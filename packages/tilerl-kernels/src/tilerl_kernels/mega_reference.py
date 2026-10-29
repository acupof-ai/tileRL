"""CPU instruction interpreter for the persistent forward kernel."""

import torch

from . import reference


def interpret_instruction(row, buffers, position):
    op, rows, cols, red, xi, yi, wi, si, oi, ai, a2, mode = map(int, row)
    x, y, w = buffers[xi], buffers[yi], buffers[wi]
    if op == 0:
        y.copy_(w[x.long()].float())
    elif op == 1:
        result = reference.rmsnorm(x.reshape(rows, cols), w, 1e-6)
        y.copy_((result.half().float() if mode else result).reshape(y.shape))
    elif op == 2:
        source = x.reshape(rows, red)
        if mode > 0:
            packed = reference.untwiddle_fp4_f16(w)
            value = reference.linear_fp4(source.half().float(), packed, buffers[si].float(), oscale=buffers[oi])
        elif mode == -1:
            weights = (reference.dequant_fp8(w, buffers[si]) * buffers[oi][:, None]).bfloat16().float()
            value = source @ weights.T
        else:
            value = source @ w.reshape(cols, red).T
        y.copy_(value.reshape(y.shape))
    elif op == 3:
        y.copy_(reference.silu_mul(x, buffers[ai]))
    elif op == 4:
        y.copy_(x + buffers[ai])
    elif op == 5:
        source = x.reshape(rows, cols)
        normed = reference.rmsnorm(source, w, 1e-6)
        y.copy_(reference.rope(normed.reshape(1, 1, rows, cols),
                               torch.tensor([position], device=x.device), 1e7,
                               rotary_dim=red).reshape(y.shape))
    elif op == 6:
        heads = a2
        query = x.reshape(rows, 1, heads, cols)
        cache = w.reshape(rows, red, mode, cols, 2)[:, :position+1]
        result = reference.dense_attention(query, cache[..., 0], cache[..., 1], cols**-0.5,
                                           q_pos=torch.tensor([position], device=x.device),
                                           k_pos=torch.arange(position+1, device=x.device))
        result *= torch.sigmoid(buffers[ai].reshape(rows, 1, heads, cols))
        y.copy_(result.reshape(y.shape))
    elif op == 7:
        heads = cols // 128
        qkv = x.reshape(rows, 1, 2*red+cols)
        z, ab, state, window = [buffers[a2+i] for i in range(4)]
        value, new_state, new_window = reference.gdn_forward(
            qkv[..., :red], qkv[..., red:2*red], qkv[..., 2*red:],
            ab[:, None, :heads], ab[:, None, heads:], state,
            z=z[:, None], conv1d_weight=w, dt_bias=buffers[oi], a_log=buffers[ai],
            norm_weight=buffers[si], conv_window=window[:, position % 2])
        y.copy_(value.reshape(y.shape))
        state.copy_(new_state)
        window[:, 1-position % 2].copy_(new_window)
    elif op == 8:
        source = x.reshape(rows, red)
        c = torch.arange(cols, device=x.device)
        index = (c // mode) * (2*mode) + c % mode + a2 if mode else c+a2
        y.copy_(source[:, index].reshape(y.shape))
    elif op == 9:
        y.reshape(rows, red, cols, 2)[:, position, :, mode].copy_(x.reshape(rows, cols))
    else:
        raise ValueError(f"Invalid opcode: {op}")


def interpret_program(program, buffers, position):
    for row in program:
        interpret_instruction(row, buffers, position)
