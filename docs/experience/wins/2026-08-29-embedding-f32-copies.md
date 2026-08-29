# The embedding table was resident three times — 9.4 GiB of a 46 GiB train step

## Context

A 27B LoRA step peaks at 47.0 / 50.6 / 57.5 / 76.5 GB at 1x64 / 1x128 / 1x256 /
2x256, and 4x256 OOMs. Those four points fit a line with slope 0.055 GiB/token
and a **~28.6 GiB intercept** — a cost independent of sequence length, so not
activations, and larger than anything the full-fine-tuning ledger could save.

`scripts/probe_train_ledger.py` prints per-phase allocation plus the largest
live CUDA storages labelled by the param key that owns them. At the backward
peak, three of the top four were the same [248320, 5120] shape:

```
  4.736 GiB  float32   (248320, 5120)   -
  4.736 GiB  float32   (248320, 5120)   -
  2.368 GiB  bfloat16  (248320, 5120)   embed_tokens
  1.184 GiB  float8_e4m3fn (248320, 5120)  lm_head.w8
```

Two 4.7 GiB f32 tensors owned by no parameter. Shape alone could not tell
`embed_tokens` from `lm_head` — the attribution is what identified them.

## What Worked

**1. The gather kernel demanded an f32 table.** `Backend.embedding` called
`_const_f32(table)`, which caches its cast, so the 27B's bf16 embedding table
lived as a permanent 4.7 GiB f32 copy beside the 2.4 GiB original. A gather
does no arithmetic — reading bf16 is exact. `make_embedding` now has a bf16
body (`T.cast` on the load) and `Backend.embedding` picks it. The C target
cannot codegen bfloat16 (`Cannot convert type bfloat16 to C type`), so CPU and
Metal keep the f32 path; this is CUDA-only.

The two bodies are written out rather than parametrized by a `dtype` argument:
`kernels.py` runs under `from __future__ import annotations`, so a `T.Tensor`
annotation is a string tilelang evaluates against module globals, and a closure
variable inside one raises `NameError`.

**2. The frozen table's gradient was computed and thrown away.** `_embedding`'s
backward returns a DENSE [vocab, hidden] f32 — another 4.7 GiB — every step,
for a table LoRA never updates. `Tape.backward` gained `needs`: the ids of the
leaves the caller will read. A handler that can skip an expensive gradient for
an unwanted leaf declares `wants = True` and receives a predicate. Only
`_embedding` uses it today; the mechanism is general.

Both are memory at the PEAK, not after it — which is the difference between
this and the streaming release rejected the same day
([errors/2026-08-29-streaming-grad-release-no-peak-win.md](../errors/2026-08-29-streaming-grad-release-no-peak-win.md)).

## Measured (27B LoRA, H20 GPU 7, `bench_harness --suite train`)

| B x T | tok/s before | after | peak GB before | after |
|---|---:|---:|---:|---:|
| 1x64 | 50.3 | **51.1** | 47.0 | **42.3** |
| 1x128 | 80.5 | **81.4** | 50.6 | **45.8** |
| 1x256 | 113.5 | **115.2** | 57.5 | **52.9** |
| 2x256 | 178.2 | **182.3** | 76.5 | **67.6** |

Every row improves on both axes; 2x256 raises its baseline. Probe phases at
1x64: forward live 31.79 -> 27.06 GiB, backward peak 46.03 -> 41.37 GiB (the
probe calls `backward` without `needs`, so its number reflects fix 1 only).
4x256 still OOMs.

## Rule

Fit the peaks against the axis they should scale with. A term that does not
move with sequence length is not activations, and finding out what it IS costs
one probe that labels storages by owner — shape alone could not separate
`embed_tokens` from `lm_head` here, and both wrong guesses were plausible.
