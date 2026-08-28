# What closing the prefill gap would actually require — 2026-08-29

Prefill ends the day at **2109.7 tok/s** (GPU-busy 970.7 ms for 2048 tokens,
64 layers, B=1). The comparison target is sglang's **4022 tok/s at the same
B=1** (`2026-08-28-vs-sglang-h20.md`; its 4908 is the B=8 row, and it runs a
dequantized bf16 checkpoint that scores 0/1000 on MMLU — same shapes, broken
model).

Reaching 4022 means 2048/4022 = **509 ms**, so 462 ms of the current 971 has to
go. Where it currently sits:

| | ms | share |
|---|---:|---:|
| `gdn_chunk_fused` | 323 | 33.3% |
| `linear_fp4_fp8_split` + `linear_fp8` | 528 | 54.4% |
| everything else | 119 | 12.3% |

**Deleting the GDN kernel entirely — zero time, physically impossible — leaves
648 ms, i.e. 3160 tok/s.** Still short. The target additionally requires
**1.36x on the two GEMMs**, which already run at 59% of the fp8 peak on their
best shapes after three rounds of tuning (N-tile 64, K-split 2, native fp8).

So the gap is not one missing optimization. It is a near-perfect linear-attention
kernel *and* a third of the remaining GEMM headroom, together.

## The one lever that has measured positive

GDN is SM-limited: 2x the blocks in one launch costs 1.20x the time
(`scripts/occ_gdn.py`). A V split is worth **1.67x on the kernel, +13.4% on
prefill** — 2110 -> ~2393, which is 1.68x from the target rather than 1.91x.
Real, worth building, and not sufficient.

Its prerequisite — moving the gated RMSNorm and z-gate out of the kernel —
compiles (measured today) but currently fails the full-scale parity gate and
`test_gdn_chunk_matches_decode`, because the decode kernel still fuses its own
epilogue. That is the next unit of work, and it is a refactor of both kernels,
not a tweak.

## What has been ruled out, with measurements

- Chunkwise-WY, three implementations: 2.6x slower, 10% slower + numerically
  wrong at scale, and 7.1x slower without compiling. The arithmetic is not the
  bottleneck — the per-token dependency chain is, and chunkwise does strictly
  less arithmetic while losing.
- The output RMSNorm as the per-step cost: 3-15%, not the 86% cycle arithmetic
  predicted.
- Prefill graph capture: ~1%, reverted. Prefill's dispatch is already hidden
  behind its GPU time.
- Speculative decoding: 0.43-0.76x of plain graph decode at every batch and
  depth measured.

## What actually moved today

Two host-side findings the per-kernel GPU table could not show, worth **1.15x**
together: 971 synchronous pageable H2D copies per prefill, and lm_head running
over all 512 positions of a chunk to use one.
