"""One-process two-forward determinism test.

Runs the model forward twice with identical inputs (fresh zeroed recurrent state
each time) and bit-compares the logits. Different -> the forward kernels are
nondeterministic (atomics / split-K reduction order). Same -> the forward is
deterministic in-process and the run-to-run divergence is cross-process
(graph capture, allocator, card state).

Pod-only: needs the 27B and a GPU. Card via CUDA_VISIBLE_DEVICES.
"""
import os
import sys

os.environ.setdefault("TILERL_QWEN38_SOURCE", "/work/Qwen3.8-27B-NVFP4")
sys.path.insert(0, "/work/tilerl-s-probestats/src")

import torch
from tilerl_kernels.backend import get_backend

from tilerl.cli import _build_model
from tilerl.engine import build_engine
from tilerl.kv_cache import BatchKv, NoPrefixStore

dev = "cuda"
backend = get_backend()
cfg, model = _build_model("qwen38-27b", seed=0, keep_master=False)
engine = build_engine(cfg, model, backend, num_slots=16, max_batch=8,
                      num_blocks=64, max_total_tokens=8192,
                      decode_graph=False, prefix_store=NoPrefixStore())

B, W = 8, 1
ids = torch.zeros(B, W, dtype=torch.int32, device=dev)
pos = torch.zeros(B, W, dtype=torch.int32, device=dev)
bt = torch.zeros(B, engine._kv.num_blocks, dtype=torch.int32, device=dev)
sl = torch.full((B,), W, dtype=torch.int32, device=dev)
sql = torch.full((B,), W, dtype=torch.int32, device=dev)
for i in range(B):
    bt[i, 0] = engine._kv.alloc_block()
# Two sets of state slots: each forward gets fresh zeroed recurrent state.
ss1 = torch.tensor([engine._states.alloc_slot() for _ in range(B)],
                   dtype=torch.int32, device=dev)
ss2 = torch.tensor([engine._states.alloc_slot() for _ in range(B)],
                   dtype=torch.int32, device=dev)


def fwd(ss):
    kv = BatchKv(block_table=bt, seq_len=sl, state_slot=ss,
                 kv_pool=engine._kv, state_pool=engine._states, seq_q_lens=sql)
    torch.cuda.synchronize()
    out = model.forward(ids, pos, kv, backend, last_only=True)
    torch.cuda.synchronize()
    return out


l1 = fwd(ss1)
l2 = fwd(ss2)
print(f"shape: {tuple(l1.shape)}  dtype: {l1.dtype}")
print(f"bitwise equal: {torch.equal(l1, l2)}")
print(f"max abs diff: {(l1.float() - l2.float()).abs().max().item():.3e}")
print(f"nonzero diff rows: {(l1 != l2).any(-1).sum().item()}/{B}")
