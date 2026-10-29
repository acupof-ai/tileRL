"""Bind one model's weights and state to the persistent instruction kernel."""

import torch

from .kernels_mega import ADD, ATTENTION, EMBED, GDN, LINEAR, NORM, ROPE, SILU, make_megakernel


class MegaForward:
    def __init__(self, model, backend, batch=1, max_context=512, cpu_reference=False,
                 blocks=None, min_blocks=8):
        cfg = model.cfg
        if backend.arch != "sm70" and not cpu_reference:
            raise ValueError("MegaForward requires sm70 or its CPU interpreter")
        self.cpu_reference = cpu_reference
        if cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128:
            raise ValueError("GDN state requires head dimension 128")
        if cfg.head_dim > 256 or cfg.rope_theta != 1e7 or cfg.rms_eps != 1e-6:
            raise ValueError("Unsupported head, RoPE, or RMSNorm configuration")
        if not cfg.full_attn_gated or cfg.tie_word_embeddings:
            raise ValueError("Requires a gated, untied model")
        self.cfg, self.backend = cfg, backend
        # 8 blocks/SM is where the cooperative grid stops growing on sm70; the
        # register cap that buys it costs GDN spills and nothing else (4 ms).
        self.blocks = blocks or backend._sms * 8
        self.min_blocks = min_blocks
        self.batch, self.max_context = batch, max_context
        self.buffers = [torch.zeros(1, device=backend.device)]
        self.rows = []
        self.weight_ids = {}
        self.state_buffers = []
        self.position = torch.zeros(1, dtype=torch.int32, device=backend.device)
        self.ids = torch.zeros(batch, dtype=torch.int32, device=backend.device)
        self.cursor = 0
        params = dict(model.params)
        if cpu_reference:
            from .reference import twiddle_fp4_f16
            for name, tensor in list(params.items()):
                if name.endswith(".wq") and getattr(tensor, "_tl_layout", "natural") == "natural":
                    params[name] = twiddle_fp4_f16(tensor)
                    params[name]._tl_layout = "tw-f16"

        def bind(tensor):
            self.buffers.append(tensor)
            return len(self.buffers) - 1

        def empty(*shape, dtype=torch.float32, state=False):
            tensor = torch.zeros(shape, dtype=dtype, device=backend.device)
            if state:
                self.state_buffers.append(tensor)
            return bind(tensor)

        def weight(name, dtype=torch.float32):
            key = (name, dtype)
            if key not in self.weight_ids:
                self.weight_ids[key] = bind(params[name].to(device=backend.device, dtype=dtype).contiguous())
            return self.weight_ids[key]

        def emit(op, r, c, k=0, x=0, y=0, w=0, s=0, o=0, a=0, a2=0, mode=0):
            self.rows.append([op, r, c, k, x, y, w, s, o, a, a2, mode])

        h, inter = cfg.hidden_size, cfg.intermediate_size
        x = empty(batch, h)
        normalized = empty(batch, h)
        attended = empty(batch, h)
        gate = empty(batch, inter)
        up = empty(batch, inter)
        activated = empty(batch, inter)
        down = empty(batch, h)
        ids_id = bind(self.ids)
        emit(EMBED, batch, h, cfg.vocab_size, ids_id, x, weight("embed_tokens", torch.float16))

        def linear(name, source, dest, n, k):
            if name + ".wq" in params:
                w = params[name + ".wq"]
                if getattr(w, "_tl_layout", "natural") != "tw-f16":
                    raise ValueError(f"Weight is not in the sm70 layout: {name}")
                if k % 512:
                    raise ValueError(f"FP4 reduction must be divisible by 512: {name}")
                scale = params[name + ".scale"]
                emit(LINEAR, batch, n, k, source, dest,
                     weight(name + ".wq", torch.uint8), weight(name + ".scale", torch.float16),
                     weight(name + ".oscale"), empty(batch, k, dtype=torch.float16), mode=k // scale.shape[1])
            elif name + ".w8" in params:
                osc = params.get(name + ".oscale", torch.ones(n, device=backend.device))
                emit(LINEAR, batch, n, k, source, dest,
                     weight(name + ".w8", torch.float8_e4m3fn), weight(name + ".wscale"),
                     bind(osc.to(backend.device).float()), mode=-1)
            elif name in params:
                emit(LINEAR, batch, n, k, source, dest, weight(name))
            else:
                raise ValueError(f"Unsupported weight representation: {name}")

        for layer in range(cfg.num_layers):
            p = f"layers.{layer}"
            emit(NORM, batch, h, x=x, y=normalized, w=weight(p + ".input_norm"), mode=1)
            if cfg.is_full_attn(layer):
                nq, nk, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
                qraw, q, key, value, qn, kn, qgate, result = (
                    empty(batch, size) for size in (2*nq*d, nq*d, nk*d, nk*d, nq*d, nk*d, nq*d, nq*d)
                )
                if p + ".qkv.wq" in params or p + ".qkv.w8" in params or p + ".qkv" in params:
                    full = empty(batch, (2*nq + 2*nk)*d)
                    linear(p + ".qkv", normalized, full, (2*nq + 2*nk)*d, h)
                    emit(8, batch, 2*nq*d, (2*nq+2*nk)*d, full, qraw)
                    emit(8, batch, nk*d, (2*nq+2*nk)*d, full, key, a2=2*nq*d)
                    emit(8, batch, nk*d, (2*nq+2*nk)*d, full, value, a2=(2*nq+nk)*d)
                else:
                    linear(p + ".q_proj", normalized, qraw, 2*nq*d, h)
                    linear(p + ".k_proj", normalized, key, nk*d, h)
                    linear(p + ".v_proj", normalized, value, nk*d, h)
                emit(8, batch, nq*d, 2*nq*d, qraw, q, mode=d)
                emit(8, batch, nq*d, 2*nq*d, qraw, qgate, a2=d, mode=d)
                emit(ROPE, batch*nq, d, cfg.effective_rotary_dim, q, qn, weight(p + ".q_norm"))
                emit(ROPE, batch*nk, d, cfg.effective_rotary_dim, key, kn, weight(p + ".k_norm"))
                cache = empty(batch, max_context, nk, d, 2, state=True)
                emit(9, batch, nk*d, max_context, kn, cache, mode=0)
                emit(9, batch, nk*d, max_context, value, cache, mode=1)
                emit(ATTENTION, batch, d, max_context, qn, result, cache, a=qgate, a2=nq, mode=nk)
                linear(p + ".o_proj", result, attended, h, nq*d)
            else:
                qd, vd, nvh = cfg.linear_q_dim, cfg.linear_v_dim, cfg.linear_num_value_heads
                qkv, z, ab, result = (empty(batch, size) for size in (2*qd+vd, vd, 2*nvh, vd))
                if p + ".qkvz.wq" in params or p + ".qkvz.w8" in params or p + ".qkvz" in params:
                    full = empty(batch, 2*qd+2*vd)
                    linear(p + ".qkvz", normalized, full, 2*qd+2*vd, h)
                    emit(8, batch, 2*qd+vd, 2*qd+2*vd, full, qkv)
                    emit(8, batch, vd, 2*qd+2*vd, full, z, a2=2*qd+vd)
                else:
                    linear(p + ".in_proj_qkv", normalized, qkv, 2*qd+vd, h)
                    linear(p + ".in_proj_z", normalized, z, vd, h)
                if p + ".ab.wq" not in params and p + ".ab" not in params:
                    raise ValueError("GDN requires fused ab projections")
                linear(p + ".ab", normalized, ab, 2*nvh, h)
                state = empty(batch, nvh, 128, 128, state=True)
                windows = empty(batch, 2, 3, 2*qd+vd, state=True)
                base = len(self.buffers)
                self.buffers.extend([self.buffers[i] for i in (z, ab, state, windows)])
                emit(GDN, batch, vd, qd, qkv, result, weight(p + ".conv1d"),
                     weight(p + ".gdn_norm"), weight(p + ".dt_bias"), weight(p + ".a_log"), base)
                linear(p + ".out_proj", result, attended, h, vd)
            emit(ADD, batch, h, x=x, y=x, a=attended)
            emit(NORM, batch, h, x=x, y=normalized, w=weight(p + ".post_attn_norm"), mode=1)
            if p + ".gate_up.wq" in params or p + ".gate_up.w8" in params or p + ".gate_up" in params:
                gu = empty(batch, 2*inter)
                linear(p + ".gate_up", normalized, gu, 2*inter, h)
                emit(8, batch, inter, 2*inter, gu, gate)
                emit(8, batch, inter, 2*inter, gu, up, a2=inter)
            else:
                linear(p + ".gate_proj", normalized, gate, inter, h)
                linear(p + ".up_proj", normalized, up, inter, h)
            emit(SILU, batch, inter, x=gate, y=activated, a=up)
            linear(p + ".down_proj", activated, down, h, inter)
            emit(ADD, batch, h, x=x, y=x, a=down)
        emit(NORM, batch, h, x=x, y=normalized, w=weight("final_norm"), mode=1)
        self.logits = torch.empty(batch, cfg.vocab_size, device=backend.device)
        linear("lm_head", normalized, bind(self.logits), cfg.vocab_size, h)
        self.program = torch.tensor(self.rows, device=backend.device, dtype=torch.int64)
        self.pointers = torch.tensor([p.data_ptr() for p in self.buffers], device=backend.device, dtype=torch.uint64)
        self.kernel = None if cpu_reference else make_megakernel(
            self.blocks, len(self.rows), len(self.buffers),
            min_blocks=self.min_blocks)

    def reset(self):
        for tensor in self.state_buffers:
            tensor.zero_()
        self.position.zero_()
        self.cursor = 0

    def prefill(self, ids):
        if ids.ndim != 2 or ids.shape[0] != self.batch or ids.shape[1] < 1:
            raise ValueError("Prompt IDs must have shape [batch, tokens]")
        if self.cursor + ids.shape[1] > self.max_context:
            raise ValueError("Maximum context exceeded")
        if self.cpu_reference:
            for token in ids.T:
                self.step(token)
            return self.logits
        tokens = ids.to(self.backend.device, torch.int32).contiguous()
        pointers = self.pointers.clone()
        pointers[self.rows[0][4]] = tokens.data_ptr()
        self.position.fill_(self.cursor)
        kernel = make_megakernel(self.blocks, len(self.rows), len(self.buffers), ids.shape[1],
                                 min_blocks=self.min_blocks)
        kernel(self.program, pointers, self.position)
        self.cursor += ids.shape[1]
        return self.logits

    def step(self, ids):
        if self.cursor >= self.max_context:
            raise ValueError("Maximum context exceeded")
        if tuple(ids.shape) != (self.batch,):
            raise ValueError("Token IDs must have one token per batch row")
        self.ids.copy_(ids)
        self.position.fill_(self.cursor)
        if self.cpu_reference:
            from .mega_reference import interpret_program
            interpret_program(self.rows, self.buffers, self.cursor)
        else:
            self.kernel(self.program, self.pointers, self.position)
        self.cursor += 1
        return self.logits
