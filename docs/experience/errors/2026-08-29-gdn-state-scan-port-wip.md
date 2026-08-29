# GDN state-scan port: right speed, wrong arithmetic — 2026-08-29

> Status: UNFINISHED, parked. `make_gdn_state_scan` exists, is not registered
> and is not dispatched; nothing in the shipped path changes.

## Why this piece

`gdn_chunk_fused` (serial scan) is 27.6% of prefill and reads 0.13% of the
tensor pipe. fla's chunked form does the same layer in 6.8 ms against our 63,
but end to end it LOSES 5% because Triton's per-launch cost puts 62.3 ms into
the gaps between kernels where ours spends 4.3
([wins/2026-08-29-fla-gdn-is-9x.md](../wins/2026-08-29-fla-gdn-is-9x.md)).
Porting to TileLang keeps the algebra and pays our 1.12 us gap instead of
Triton's 9.02 — worth **1.19x on prefill, 2232 -> ~2660 tok/s**.

Of the four stages, exactly one had ever lost: the two 2026-08-25 WY rejections
had a chunk interior 2.9x FASTER than the serial kernel and an inter-chunk state
scan 285x slower than fla's. So this port is that one kernel.

## Where it got to

Speed is there: **34.5 us a layer against fla's 34.0 (0.99x)** at our shapes.

Arithmetic is not. The carried state never reaches the accumulator, so the
kernel computes `S_next = K^T V_new` with the `e_last * S` term missing.
Known-answer cases (`scripts/probe_scan_mini.py`, one chunk so no cross-chunk
carry is involved):

| case | expect | got |
|---|---:|---:|
| all zero | 0 | 0 |
| W=K=0, U=1, S=1 | 1 | **0** |
| + K=1 | 65 | **64** |
| K=U=0, any gate | `exp2(G_last)` | **0** |

Both wrong rows are short by exactly the initial state.

## What it is not

- **Not `clear_accum`.** Its default is `False`, so the gemm should accumulate.
- **Not the gate scalar, and not variable shadowing.** I did find that
  rebinding a `T.alloc_var` with a Python `=` swaps the device scalar for a
  host-side expression, and split it into `g_last` / `g_decay` — no change.
- **Not the elementwise decay pass.** Bisected: disabling the multiply entirely
  leaves every number identical. So the fault is `T.gemm` not accumulating into
  `h_fr`'s existing value.

## Next

The reference keeps `T.copy(b_h_shared, h[...])` at the TOP of its chunk loop —
it exports the per-chunk state, which we do not need and which I dropped. That
read is the difference between its loop and mine, and is a candidate for what
keeps `b_h_fragment` live across the gemm in tilelang's eyes. Restore it (or
route the state through shared every iteration) and re-run the mini probe.

## Rule

Bisect by DELETING the suspect, not by rewriting it. Two fixes aimed at the
gate scalar changed nothing; one deletion of the whole decay pass proved in a
single run that the scalar was never involved. The probe that says "same
numbers with this code removed" is worth more than two plausible repairs.

Also: my own probe was the bug twice today — `G` must be a chunk-local
cumsum and I fed it a constant, which produced 64/23.5/3.18 and no way to read
them. A probe needs its inputs checked as carefully as the kernel it tests.
