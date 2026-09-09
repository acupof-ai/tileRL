"""Pool-poison determinism test: does the decode forward read an address no
kernel in the captured sequence writes?

Two captures, same inputs, same process, differing ONLY in a poison byte
written inside the capture before the forward runs:
  A: poison 0x00  B: poison 0xFF
A big tensor is allocated, filled, and freed INSIDE the capture; the forward's
intermediates then reuse that memory (pool_reuse_probe.py: within-capture
free+reuse is 1.0000; cross-capture reuse is 0.0000, so a separate filler
capture cannot poison the next capture). If the replay logits differ, a kernel
reads a pool address that no kernel in the sequence writes — the cross-process
nondeterminism has its mechanism. If bitwise equal, every read address is
written and the uninitialized-memory hypothesis is dead.

Pod-only: needs the 27B and a GPU. Card via CUDA_VISIBLE_DEVICES.
"""
import os
import sys

os.environ.setdefault("TILERL_QWEN38_SOURCE", "/work/Qwen3.8-27B-NVFP4")
sys.path.insert(0, "/work/tilerl-s-probestats/src")

import torch

from tilerl.cli import _build_model
from tilerl.engine import build_engine
from tilerl.kv_cache import BLOCK_TOKENS, BatchKv, NoPrefixStore
from tilerl_kernels.backend import get_backend

dev = "cuda"
backend = get_backend()
cfg, model = _build_model("qwen38-27b", seed=0, keep_master=False)
engine = build_engine(cfg, model, backend, num_slots=16, max_batch=8,
                      num_blocks=64, max_total_tokens=8192,
                      decode_graph=False, prefix_store=NoPrefixStore())

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
for i in range(B):
    for j in range(NB):
        bt[i, j] = engine._kv.alloc_block()
ss = torch.tensor([engine._states.alloc_slot() for _ in range(B)],
                  dtype=torch.int32, device=dev)
pkv = BatchKv(block_table=bt, seq_len=psl, state_slot=ss,
              kv_pool=engine._kv, state_pool=engine._states, seq_q_lens=psql)
dkv = BatchKv(block_table=bt, seq_len=dsl, state_slot=ss,
              kv_pool=engine._kv, state_pool=engine._states,
              seq_q_lens=torch.full((B,), W, dtype=torch.int32, device=dev))

pool = torch.cuda.graph_pool_handle()

# The decode forward's intermediates are < 256 MiB at B=8; the poison tensor
# is freed inside the capture and the forward's allocations split from it.
POISON_SIZE = 1 << 30
READER_SIZE = 256 << 20


def reset_inputs():
    """Zero the recurrent state and re-prefill, so the next forward reads the
    same K/V and state as a fresh decode. Indexed assignment, not .zero_() on
    an advanced-indexing view (states[ss] is a copy)."""
    engine._states.states[ss] = 0
    if engine._states.conv_windows is not None:
        engine._states.conv_windows[ss] = 0
    engine._states.win_parity[ss] = 0
    torch.cuda.synchronize()
    model.forward(pids, ppos, pkv, backend, last_only=True)
    torch.cuda.synchronize()


def capture_forward(poison_value):
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, pool=pool):
        p = torch.empty(POISON_SIZE, dtype=torch.uint8, device=dev)
        p.fill_(poison_value)
        del p  # free inside the capture; the forward reuses this memory
        logits = model.forward(did, dpos, dkv, backend, last_only=True)
    return gr, logits


def poisoned_logits(poison_value):
    reset_inputs()
    g, logits = capture_forward(poison_value)
    cap = logits.clone()
    reset_inputs()
    g.replay()
    torch.cuda.synchronize()
    return cap, logits.clone()


# Warmup: 2 eager forwards (JIT), like _DecodeGraph.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        model.forward(did, dpos, dkv, backend, last_only=True)
torch.cuda.current_stream().wait_stream(s)

# --- Positive control: within-capture poison reaches a same-capture read ---
def positive_control(value):
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, pool=pool):
        p = torch.empty(POISON_SIZE, dtype=torch.uint8, device=dev)
        p.fill_(value)
        del p
        t = torch.empty(READER_SIZE, dtype=torch.uint8, device=dev)
        out = t.clone()
        del t
    gr.replay()
    torch.cuda.synchronize()
    return (out == value).all().item()


ok0 = positive_control(0x00)
ok1 = positive_control(0xFF)
print(f"positive control: poison(0x00) -> reader all-zero: {ok0}")
print(f"positive control: poison(0xFF) -> reader all-0xFF:  {ok1}")
if not (ok0 and ok1):
    print("BLIND: within-capture poison does not reach a same-capture read; aborting")
    sys.exit(1)

# --- Real test: 0x00-poison vs 0xFF-poison forward captures ----------------
cap0, rep0 = poisoned_logits(0x00)
cap1, rep1 = poisoned_logits(0xFF)

print("=== replay logits: 0x00-poison vs 0xFF-poison ===")
print(f"bitwise equal: {torch.equal(rep0, rep1)}")
print(f"max abs diff: {(rep0.float() - rep1.float()).abs().max().item():.3e}")
print(f"nonzero diff rows: {(rep0 != rep1).any(-1).sum().item()}/{B}")

print("=== capture-time logits (bonus) ===")
print(f"bitwise equal: {torch.equal(cap0, cap1)}")
print(f"max abs diff: {(cap0.float() - cap1.float()).abs().max().item():.3e}")

# --- Same-poison control: the harness must be deterministic ---------------
cap0b, rep0b = poisoned_logits(0x00)
print("=== same-poison control (0x00 vs 0x00) ===")
print(f"bitwise equal: {torch.equal(rep0, rep0b)}")
print(f"max abs diff: {(rep0.float() - rep0b.float()).abs().max().item():.3e}")
