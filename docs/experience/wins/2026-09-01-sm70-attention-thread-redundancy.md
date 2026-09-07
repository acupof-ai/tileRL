# sm70 split attention: the context slope was thread redundancy, not bandwidth — 2026-09-01

## Context

Decode on the V100 fit `ms/tok = 31.9 + 6.20 * (ctx/1K)`, residuals within
±0.3 ms from 32 to 4096 — a clean line whose slope had no physical basis. KV is
128 KiB/token (16 full-attn layers × 4 KV heads × **256** dim × 4 B × 2), so 1K of
context is 134 MB = **0.15 ms** at 900 GB/s. The measured slope was **42× off
roofline**, and it capped long-context decode at 17.4 tok/s against 32.7 short.

This paragraph first read 64 KiB/token, 67 MB, 0.07 ms and **83×**, from a
`head_dim` of 128. The checkpoint's `text_config.head_dim` is **256**
(`num_key_value_heads` 4, `num_hidden_layers` 64), so every byte figure here was
half the real one. The conclusion is unaffected — 42× is as far off roofline as
83×, and the cause was identified by timing the two kernels, not by the ratio —
but the halved bytes propagate: any later reasoning that starts from
"64 KiB/token" on sm70 is using bf16's byte count on a pool that
`backend.py:353` allocates as **f32** for `arch in ("cpu", "metal", "sm70")`,
routed as `kv_io` at `engine.py:1513`/`:1529`.

**This entry already knew.** The note at the µs tables below records the same
`head_dim` 128 error, and the third Rule states it as a lesson. The byte figure
in this paragraph is the one place the correction was never applied — the fix
went to the benchmark harness that produced the tables, and the arithmetic in
the prose kept the old constant for six days. A rule written from a defect does
not retroactively check the file it is written in.

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
83% of the 56.1 tok/s *dense* weight-bandwidth roofline, and **34.3% of the
ceiling a speculative rate is bound by** — at 2.9 tok/forward over a 19.24 GB
depth-3 tick a weight bound on forwards permits 135.7 tok/s. The draft forward's
1.065 GB was measured 2026-09-06, replacing an earlier 29-37% bracket
(`errors/2026-09-06-a-spec-rate-over-a-dense-roofline.md`).

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

Fourth, from that same constant surviving six days in this file: **a
wrong-constant sweep has to cover the prose in the same file, not only the code
that produced the numbers.** The 2026-09-01 fix went to the harness and to the
µs tables; the Context section's arithmetic kept `head_dim` 128 and quietly
halved every byte figure derived from it, six days after the Rule above was
written here. A rule written from a defect does not retroactively check the file
it is written in — the sweep is a separate act, and its scope is the figure, not
the factorisation: two different wrong operands (dim 128 with f32 here, dim 256
with bf16 in a later break-even) both produced the same memorable 64 KiB/token.
