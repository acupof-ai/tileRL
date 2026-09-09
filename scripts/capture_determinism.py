"""Same-process two-capture determinism test, with input reset.

The first version was contaminated: the decode forward ADVANCES the gated-delta
recurrent state (a per-slot mutable tensor, not routed through the block table),
so replay 2 read a different state than replay 1. The 5.965 / 8.367 diffs were
state advancement, not capture nondeterminism.

This version resets the recurrent state and re-runs the prefill before every
replay, so each replay reads bitwise-identical inputs. It also dumps the full
input set before replay A and replay B and bit-compares them, per the directive
to prove the inputs are the same before blaming the capture.

  two replays equal -> capture + replay are deterministic in-process; the
    cross-process divergence is the environment.
  two replays differ -> a real choice in capture/replay.

Pod-only: needs the 27B and a GPU. Card via CUDA_VISIBLE_DEVICES.
"""
import os
import sys

os.environ.setdefault("TILERL_QWEN38_SOURCE", "/work/Qwen3.8-27B-NVFP4")
sys.path.insert(0, "/work/tilerl-s-probestats/src")

import torch

from tilerl.cli import _build_model
from tilerl.engine import _DecodeGraph, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, BatchKv, NoPrefixStore
from tilerl_kernels.backend import get_backend

dev = "cuda"
backend = get_backend()
cfg, model = _build_model("qwen38-27b", seed=0, keep_master=False)
engine = build_engine(cfg, model, backend, num_slots=16, max_batch=8,
                      num_blocks=64, max_total_tokens=8192,
                      decode_graph=True, prefix_store=NoPrefixStore())

B, W = 8, 1
PW = 48  # prefill tokens

g = torch.Generator(device=dev).manual_seed(1234)
did = torch.randint(100, 2000, (B, W), generator=g, dtype=torch.int32, device=dev)
dpos = torch.full((B, W), PW, dtype=torch.int32, device=dev)
dsl = torch.full((B,), PW + 1, dtype=torch.int32, device=dev)
pids = torch.randint(100, 2000, (B, PW), generator=g, dtype=torch.int32, device=dev)
ppos = torch.arange(PW, dtype=torch.int32, device=dev).unsqueeze(0).expand(B, PW).contiguous()
psl = torch.full((B,), PW, dtype=torch.int32, device=dev)
psql = torch.full((B,), PW, dtype=torch.int32, device=dev)

NB = (PW + BLOCK_TOKENS - 1) // BLOCK_TOKENS + 1
bt = torch.zeros(B, engine._kv.num_blocks, dtype=torch.int32, device=dev)
blocks_flat = []
for i in range(B):
    for j in range(NB):
        b = engine._kv.alloc_block()
        bt[i, j] = b
        blocks_flat.append(b)
blocks_flat = torch.tensor(blocks_flat, dtype=torch.int64, device=dev)
ss = torch.tensor([engine._states.alloc_slot() for _ in range(B)],
                  dtype=torch.int32, device=dev)
pkv = BatchKv(block_table=bt, seq_len=psl, state_slot=ss,
              kv_pool=engine._kv, state_pool=engine._states, seq_q_lens=psql)

pool = torch.cuda.graph_pool_handle()
gA = _DecodeGraph(model, backend, engine._kv, engine._states, B, width=W, pool=pool)
gB = _DecodeGraph(model, backend, engine._kv, engine._states, B, width=W, pool=pool)


def reset_inputs():
    """Zero the recurrent state and re-run the prefill, so the next replay reads
    the same K/V and state as the first. The decode overwrites position PW's K/V
    before the attention reads it, so only the state and positions 0..PW-1 need
    resetting; the prefill rewrites those. Indexed ASSIGNMENT, not .zero_() on
    an advanced-indexing view: states[ss] is a copy, so .zero_() would zero the
    copy and leave the live state advanced from the last replay."""
    engine._states.states[ss] = 0
    if engine._states.conv_windows is not None:
        engine._states.conv_windows[ss] = 0
    engine._states.win_parity[ss] = 0
    torch.cuda.synchronize()
    model.forward(pids, ppos, pkv, backend, last_only=True)
    torch.cuda.synchronize()


def dump_inputs():
    """Everything the replay reads: recurrent state, K/V (+scales) for the blocks,
    and the static input buffers."""
    d = {
        "states": engine._states.states[ss].clone(),
        "win_parity": engine._states.win_parity[ss].clone(),
        "k": engine._kv.k_pool[:, blocks_flat, :, :, :].clone(),
        "v": engine._kv.v_pool[:, blocks_flat, :, :, :].clone(),
        "ids": did.clone(), "pos": dpos.clone(), "bt": bt.clone(),
        "sl": dsl.clone(), "ss": ss.clone(),
    }
    if engine._states.conv_windows is not None:
        d["conv"] = engine._states.conv_windows[ss].clone()
    if engine._kv.k_scale is not None:
        d["k_scale"] = engine._kv.k_scale[:, blocks_flat, :, :].clone()
        d["v_scale"] = engine._kv.v_scale[:, blocks_flat, :, :].clone()
    return d


def run_graph(gr):
    gr._ids.copy_(did)
    gr._pos.copy_(dpos)
    gr._bt.copy_(bt)
    gr._sl.copy_(dsl)
    gr._ss.copy_(ss)
    torch.cuda.synchronize()
    gr._graph.replay()
    torch.cuda.synchronize()
    return gr._logits.clone()


def cmp_inputs(a, b):
    for k in a:
        if not torch.equal(a[k], b[k]):
            print(f"  INPUT DIFF: {k}")
            return False
    return True


reset_inputs()
dA = dump_inputs()
la = run_graph(gA)

reset_inputs()
dB = dump_inputs()
lb = run_graph(gB)

print("=== input comparison (replay A vs replay B) ===")
inputs_same = cmp_inputs(dA, dB)
print(f"inputs bitwise identical: {inputs_same}")

print("=== two-capture comparison ===")
print(f"bitwise equal: {torch.equal(la, lb)}")
print(f"max abs diff: {(la.float() - lb.float()).abs().max().item():.3e}")
print(f"nonzero diff rows: {(la != lb).any(-1).sum().item()}/{B}")

# Control: same graph, reset inputs, replay twice.
reset_inputs()
la2 = run_graph(gA)
print("=== same-graph control (inputs reset) ===")
print(f"bitwise equal: {torch.equal(la, la2)}")
print(f"max abs diff: {(la.float() - la2.float()).abs().max().item():.3e}")
