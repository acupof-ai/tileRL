"""H1 probe for the lazy sparse-graph warmup scribble (V100, 2026-09-14).

The sparse decode graph is captured lazily on its first real tick. Warmup (2
forwards) and the capture forward all run on ZEROED static buffers, all
targeting state slot 0 and physical block 0. On main slot 0 is a live row:
each warmup/capture GDN forward flips that slot's conv-window parity, so 3
extra forwards invert the bank the real replay then reads — degenerate
outputs on sparse+spec W=2 ticks, only when the graph is created mid-traffic.

Decisive signal: one real decode tick flips win_parity of the row's slot
exactly ONCE. The probe forces a lazy re-capture mid-request and checks:

  main   -> MAIN_BUG: the live slot's parity (and other non-pad slots) carries
            the extra warmup flips; block/slot 0 were written by warmup
  branch -> BRANCH_OK: only the pad slot takes the warmup writes

Run (V100):
  TILERL_TARGET=cuda PYTHONPATH=src:packages/tilerl-kernels/src \
    /usr/bin/python3 scripts/probe_sparse_graph_warmup_pad.py \
      /path/to/qwen38-27b --draft /path/to/model_mtp.safetensors
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

import torch
from tilerl_kernels.backend import get_backend

from tilerl import cli
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--draft", required=True)
    args = ap.parse_args()
    cli._QWEN38_SOURCE = args.source

    be = get_backend()
    assert be.device.type == "cuda", "this probe needs the CUDA cell"
    cfg, model = _build_model("qwen38-27b", seed=0, fuse_projections=True)
    from tilerl.spec import load_draft

    e = build_engine(
        cfg, model, be, num_slots=2, max_batch=1,
        max_total_tokens=4096, max_num_batched_tokens=512,
        sparse_k=128, scorer="bounds", kv_cold_bytes=1 << 30,
        decode_graph=True, draft=load_draft(model, args.draft), spec_depth=1)
    e.run()

    ids = [(t % 31000) + 7 for t in range(1024)]
    rid = e.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=24, seed=0))
    while not e._running:
        pass
    r = next(x for x in e._running if x.req_id == rid)

    def step():
        out = e.poll().get(rid, ())
        e.step()
        return out

    while len(step()) < 1:  # cross prefill; lazily create the W=1/W=2 graphs
        pass
    assert e._sparse_graphs, "no sparse graph captured during warmup traffic"

    # Control: one normal replay tick (graphs exist) must flip parity exactly once.
    states = e._states
    p0 = states.win_parity.clone()
    step()
    torch.cuda.synchronize()
    p1 = states.win_parity.clone()
    slot = r.state_slot
    print(f"row slot={slot} pad_slot={e._pad_slot}")
    assert int(p1[slot] - p0[slot]) % 2 == 1, "control tick did not flip the row slot once"

    # Force a fresh lazy capture on the next tick (new SparseForward + warmup + capture).
    e._sparse_graphs.clear()
    p2 = states.win_parity.clone()
    step()
    torch.cuda.synchronize()
    p3 = states.win_parity.clone()

    pad_s = e._pad_slot
    victims = sorted(
        int(i) for i in range(p3.numel())
        if i != pad_s and int((p3[i] - p2[i]) % 2) == 0)  # expected exactly one flip
    print(f"slots missing their single flip after a lazy-capture tick: {victims}")
    if pad_s is not None:
        extra = int((p3[pad_s] - p2[pad_s]) % 2)
        print(f"pad slot {pad_s} flips mod 2: {extra} (warmup writes land here on the branch)")
    e.shutdown()
    if slot in victims:
        print("MAIN_BUG: warmup/capture flipped the live row slot extra times")
        sys.exit(10)
    print("BRANCH_OK: the live row slot flipped exactly once")


if __name__ == "__main__":
    main()
