# sm70 split attention: the context slope was thread redundancy, not bandwidth — 2026-09-01

## Context

Decode on the V100 fit `ms/tok = 31.9 + 6.20 * (ctx/1K)`, residuals within
±0.3 ms from 32 to 4096 — a clean line whose slope had no physical basis. KV is
64 KiB/token (16 full-attn layers × 4 KV heads × 128 dim × 4 B × 2), so 1K of
context is 67 MB = **0.07 ms** at 900 GB/s. The measured slope was **83× off
roofline**, and it capped long-context decode at 17.4 tok/s against 32.7 short.

GDN decode is O(1) in context (`kernels_gdn.py` `gdn_decode_fused` takes no
SeqLens and has no history loop — state is updated in place), so the whole
slope belonged to the 16 full-attention layers.

## What Worked

Timing the two kernels separately, which had never been done — the split-KV
change shipped on end-to-end tok/s alone, and that is how a 6× hid for a week.

`paged_attention_split` at KVSPLIT=16, per call:

| ctx | split µs | combine µs |
|---:|---:|---:|
| 512 | 129.2 | 70.6 |
| 2048 | 480.4 | 60.9 |
| 8192 | 1910.7 | 60.8 |

Split carried all of the growth. The cause showed up in a thread sweep:

| 4096 ctx, split only | 32t | 64t | 128t | 256t |
|---|---:|---:|---:|---:|
| µs | 780 | 950 | 2165 | 4066 |

**Cost rising with thread count is redundancy, not work.** The per-position dot
was `for d in T.serial(D)` — never distributed, so all 64 threads ran the same
128-step chain, each step waiting on its own global load. 32 threads was the
floor because a warp executes in lockstep. Arithmetic agrees: 4096/16 splits =
256 positions × 128 serial FMAs = 32768 dependent steps ≈ 85 µs of pure latency,
against 948 µs measured — the 11× gap is the per-step global load.

Fix: stage `block_N=16` positions into fragments, reduce with `T.reduce_sum`
over a `(block_N, D)` product, KVSPLIT 16 → 32. Parity against a torch
reference holds (max relerr 5.3e-05 at ctx 512/2048, S=1 and S=4).

| ctx | split µs | comb µs | total | was | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 97.8 | 59.0 | 156.8 | 199.8 | 1.27× |
| 1024 | 110.4 | 67.8 | 178.2 | 319.1 | 1.79× |
| 2048 | 99.4 | 59.7 | 159.1 | 541.3 | 3.40× |
| 4096 | 102.5 | 60.7 | 163.2 | 1024.0 | 6.27× |
| 8192 | 193.0 | 59.3 | 252.3 | 1971.5 | 7.81× |

The speedup column understates it. What matters is that **512 → 4096 is flat**
(157 → 163 µs): at KVSPLIT=32 a block owns ≤128 positions, below launch
overhead, so the context slope is gone rather than reduced. The thread sweep
now reads 105/102/142/304 µs — flat to 64t, so the redundancy is gone and what
remains past 128t is occupancy.

(Both columns were measured at head_dim 128. The real Qwen3.8-27B is
**head_dim 256** — `config.py:147` — so every absolute µs above is ~2×
optimistic. The dot is O(D), so the ratios stand and the diagnosis stands; the
absolute cost does not. The bench now defaults to 256 and prints the shape it
ran, which is the only reason the error surfaced.)

End to end, dense decode, steady state with no prefill in the window:

| ctx | tok/s after | before |
|---:|---:|---:|
| 32 | 32.4 | 31.1 |
| 512 | 32.1 | 28.5 |
| 1024 | 31.8 | 26.2 |
| 2048 | 31.2 | 22.6 |
| 4096 | **30.0** | **17.4** |

**1.72× at 4096.** The slope fell from 6.20 to 0.59 ms per 1K — **10.5×** — and
decay from 32 to 4096 tokens of context went from 44% to 7%.

Speculation gains more than dense does, because a verify forward multiplies the
serial attention cost by the chain width:

| ctx | spec d3 after | before | vs dense now |
|---:|---:|---:|---:|
| 32 | 32.9 | 31.8 | 1.02× |
| 512 | 44.8 | 37.7 | 1.40× |
| 1024 | **46.5** | 34.5 | 1.46× |
| 2048 | 40.8 | 24.4 | 1.31× |
| 4096 | 37.0 | 16.7 | 1.23× |

**2.22× at 4096.** Speculation had been a net LOSS there (16.7 against 17.4
dense); it is now a win at every context. tok/fwd holds at 2.9-3.3 across the
whole range, so draft acceptance never degraded with context — attention was
eating the gain the whole time. Peak throughput is **46.5 tok/s at 1024 ctx**,
73% of the 64 tok/s weight-bandwidth roofline.

Sharing the K/V tile across the GQA group would cut cache traffic another 6×,
but a `(gq, D)` fragment fails LayoutInference
(`CanProveEqual(abs(source->scale), 1)`) even padded from 6 to 8 — the 2D shape
is the blocker, not the odd group size. A compile-only probe of both candidate
shapes settled that in one 2-second job instead of a 3-minute round trip, and
the block stayed per-query-head.

## Rule

An end-to-end number cannot locate a regression inside a two-kernel pipeline.
Time each kernel when you change one — the split-KV rewrite was accepted on
tok/s alone and shipped a 6× defect that survived three later measurements.

The diagnostic that pays: **sweep thread count**. Real work is flat or falling;
cost that rises with threads means every thread is doing the same thing. That
one sweep separated "needs a better schedule" from "needs any schedule at all"
without reading a line of PTX.

Third, from the head_dim error: a microbenchmark that hardcodes the shape is
measuring a model nobody runs. Read the dimension out of `config.py` or print
what you ran — a wrong constant in the harness is invisible in every number it
produces, and this one was off by 2× on the exact axis under investigation.
