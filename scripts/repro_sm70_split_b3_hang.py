#!/usr/bin/env python3
"""Minimal offline repro for the sm70 paged_attention_split B=3 launch hang.

Standalone, read-only, one process; does not touch the server. Builds the real
cuda Backend and calls Backend.paged_attention for the prefill split path with
B in {1,2,3,4} at padded S=512 (wide KVSPLIT=16), ragged per-row query lengths,
and reports for EACH launch:

* compile-vs-device-loop: the same shape is launched REPEATS times; the first
  pays the lazy JIT compile. If launch 2+ return, B was merely un-warmed (fix =
  warm it). If launch 2+ also exceed the timeout, the specialized device kernel
  itself loops forever (fix = tilelang codegen + a B guard).
* a hard wall-clock timeout per launch, enforced from a watchdog thread, so a
  hung launch is reported rather than waited on. The CUDA context is left to
  die with the process (a hung kernel cannot be cancelled), hence each hang is
  the LAST thing we attempt; ordering runs healthy shapes first.

Run on the card:
    TILERL_TARGET=cuda python3 scripts/repro_sm70_split_b3_hang.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

# Path bootstrap mirrors scripts/v100.sh.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (f"{_ROOT}/src", f"{_ROOT}/packages/tilerl-kernels/src"):
    sys.path.insert(0, p)

S = 512              # padded query width; >=8 picks SM70_KVSPLIT_WIDE=16
H = 24               # query heads (qwen38-27b GQA: Hkv 8)
HKV = 8
D = 256              # head dim sm70 serves in f32 IO
BLOCK = 16
# Small pool keeps this standalone repro inside the few-hundred MiB free beside
# the resident 32 GB server. NB/Mb are NOT the suspected JIT axis (B,S,H,KVSPLIT
# are): the causal bound covers only prefill_from+S keys, so a 256-block pool is
# enough for first-chunk prefill and leaves the B=3 specialization identical.
NUM_BLOCKS = int(os.environ.get("NUM_BLOCKS", "256"))
LAUNCH_TIMEOUT_S = float(os.environ.get("HANG_TIMEOUT_S", "20"))
REPEATS = int(os.environ.get("REPEATS", "3"))
BATCHES = tuple(int(x) for x in os.environ.get("BATCHES", "1,2,3,4").split(","))
# ragged per-row valid query lengths, padded to S in the q tensor.
SEQ_Q = {1: [512], 2: [512, 384], 3: [512, 384, 256], 4: [512, 384, 256, 256]}


def _build_inputs(b: int, device):
    import torch

    seq_q = SEQ_Q[b]
    # f32 Q [B,S,H,D]; padded lanes past each row's valid length are garbage but
    # the kernel bounds them out (its output for tt>=SeqQLens is never read).
    q = torch.randn(b, S, H, D, dtype=torch.float32, device=device)
    k_cache = torch.randn(NUM_BLOCKS, HKV, BLOCK, D, dtype=torch.float32, device=device)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.zeros(b, NUM_BLOCKS, dtype=torch.int32, device=device)
    for i in range(b):
        # Give every row a distinct span of physical blocks covering the
        # longest history this S could address (prefill_from + S).
        block_table[i] = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
    # First-chunk prefill: prefill_from=0 -> SeqLen = SeqQLen, history 0.
    seq_lens = torch.tensor(seq_q, dtype=torch.int32, device=device)
    seq_q_lens = torch.tensor(seq_q, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, block_table, seq_lens, seq_q_lens


def _timed_launch(backend, args, result: dict, errors: list):
    try:
        torch = __import__("torch")
        q, k_cache, v_cache, block_table, seq_lens, seq_q_lens = args
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = backend.paged_attention(
            q, k_cache, v_cache, block_table, seq_lens, 1.0 / (D ** 0.5),
            seq_q_lens=seq_q_lens)
        torch.cuda.synchronize()
        result["ms"] = (time.perf_counter() - t0) * 1e3
        result["shape"] = tuple(out.shape)
    except Exception as exc:  # noqa: BLE001 - report, classify by caller
        errors.append(repr(exc))


def main() -> int:
    import torch
    from tilerl_kernels.backend import get_backend

    assert torch.cuda.is_available(), "this repro must run on the sm70 card"
    backend = get_backend()
    print(f"backend target={backend.target} arch={getattr(backend, 'arch', '?')} "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"S={S} wide-KVSPLIT=16, repeats/shape={REPEATS}, "
          f"launch timeout={LAUNCH_TIMEOUT_S}s\n")

    hanged: list[tuple] = []
    errored: list[tuple] = []
    ok_repeat2: set[int] = set()
    for b in BATCHES:
        args = _build_inputs(b, torch.device("cuda"))
        for rep in range(1, REPEATS + 1):
            result: dict = {}
            errors: list[str] = []
            th = threading.Thread(
                target=_timed_launch, args=(backend, args, result, errors), daemon=True)
            th.start()
            th.join(LAUNCH_TIMEOUT_S)
            if th.is_alive():
                verdict = f"HANG (>{LAUNCH_TIMEOUT_S:.0f}s, launch never returned)"
                hanged.append((b, rep))
                print(f"B={b} rep={rep}: {verdict}")
                # The kernel/context cannot be killed; later shapes in this
                # process are moot. Report and exit non-zero.
                print("\ndevice kernel loop confirmed -- abandoning further launches "
                      "(a hung CUDA launch is not cancellable in-process).")
                _summary(hanged, errored, ok_repeat2)
                return 2
            if errors:
                errored.append((b, rep))
                verdict = f"ERROR {errors[0][:200]}"
            else:
                if rep >= 2:
                    ok_repeat2.add(b)
                note = "incl. lazy JIT" if rep == 1 else "cached shape"
                verdict = f"ok {result['ms']:9.1f} ms out={result['shape']} ({note})"
            print(f"B={b} rep={rep}: {verdict}")
        print()

    _summary(hanged, errored, ok_repeat2)
    return 0 if not errored else 3


def _summary(hanged, errored, ok_repeat2):
    if errored:
        print(f"RESULT: {len(errored)} launch(es) ERRORED (not hang, not success): "
              f"{errored[:6]}. No B-vs-hang conclusion until these return -- free GPU "
              "(OOM beside resident server) or put cuda-12.4 on PATH (nvcc c++20).")
    elif 3 in ok_repeat2 and not hanged:
        print("RESULT: B=3 returned on repeat 2+ -> un-warmed-shape JIT only (fix = "
              "prewarm), no device dead-loop.")
    elif hanged:
        print(f"RESULT: HANG at {hanged}. Repeat-after-compile hang => device kernel "
              "dead-loop for that B; needs a B-guard fallback now + tilelang codegen fix.")
    else:
        print("RESULT: no hang, but B=3 repeat 2+ was not observed (check REPEATS/BATCHES).")


if __name__ == "__main__":
    raise SystemExit(main())
