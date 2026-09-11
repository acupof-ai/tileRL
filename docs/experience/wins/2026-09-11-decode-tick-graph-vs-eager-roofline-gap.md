# Decode tick at 12% of HBM roofline: where the 47 ms go (graph replay fixes 4.1×) — sm90 H20, 2026-09-11

> Status: **Shipped** — measured in the engine loop with the CUDA decode graph
> captured, B=1 and B=8, 32-tick steady decode windows (no prefill), Qwen3.8-27B
> NVFP4, 4096-token context, physical H20 card 6. Graph beats eager at both
> batches; the residual gap is the small-M MMA band, not launch overhead.

## Context

`bench --kernels` reported the B=1 decode `TIMED TOTAL` as **53.2 ms against a
6.6 ms HBM roofline bound — 12.4%**. A served decode tick at 19 tok/s when the
weights stream in 6.6 ms implies either the kernels are 8× off their own
roofline or the table is timing something the engine does not do. It is the
second: the table dispatches each of 497 kernels eagerly, one launch each, and
a launch costs ~0.1 ms here. The serving engine captures the decode tick as one
CUDA graph and replays it with a single launch. This entry prices that
difference in the engine loop — not from `bench --kernels` — and names what is
left.

## What worked

Four arms in one process (`/work/probe_decode_clean_b8.py`): graph vs eager ×
B=1 vs B=8, same checkpoint and backend. Each arm submits its rows together,
waits until every row is in decode, runs 8 warm ticks, then times
`engine.step()` over 32 ticks that contain **only** decoding rows — asserted,
not assumed. The graph arm asserts `(B, 1) in eng._decode_graphs`, so it can
never silently measure eager.

Engine loop, 32 decode-only ticks, 4096-token context (`ms`, median; min in
brackets):

| B | CUDA graph replay | eager per-launch | eager/graph |
|---:|---:|---:|---:|
| 1 | 11.68 (11.65) | 47.99 (47.52) | 4.11× |
| 8 | 25.98 (25.79) | 64.69 (62.85) | 2.49× |

Roofline bounds (s=4096, measured 3312.71 GB/s floor): **6.6 ms at B=1**
(22.36 GB) and **7.74 ms at B=8** (25.63 GB). The graph tick is at **57% of
bound (1.77× over) at B=1** and **30% (3.36× over) at B=8** — batching 8 rows
2.22× the bytes costs 2.22× the graph time (11.68 → 25.98 ms), so graph
decode scales with bytes even though the per-bucket launch count does not
change; the gap to the bound widens with B.

Two apparatus bugs the first runs hit — both measured before the number was
trusted:

- **The pool fit has to be asked for by name.** `build_engine(num_blocks=0)`
  fits the KV pool to free memory; passing `max_blocks=0` (the cap) leaves
  `num_blocks` at its **64-block default**, which fits ~two 4096-token rows.
  Six requests then starve in the queue and die one at a time with
  `pool_exhausted: need 1 block(s), 0 free` — surfaced through the request
  `_failed` map, not the process, so the arm fails late with "rows never all
  reached decode". The fitted B=8 pool is ~76 GiB.
- **A repeated sentence N times is not 4096 tokens.** `"…dog. " * 40`
  tokenizes to ~416 tokens, so the early B=1 rows were short-context ticks,
  not the s=4096 tick the 6.6 ms roofline row prices (weight-streaming
  dominated anyway, so the medians barely moved — 11.60–11.72 ms — which is
  exactly why context length had to be asserted rather than inferred from the
  ms). The final prompts repeat the sentence 400× and slice to exactly 4096.
- A stale `TILERL_QWEN38_SOURCE` pointing at the deleted
  `/work/Qwen3.8-27B-NVFP4` path makes `load_hf` treat the path as an HF repo
  id and raise `Repo id must be in the form …`; the restored checkpoint is
  `/work/tilerl-ckpt/Qwen3.8-27B-NVFP4`.

### Where the graph's remaining 18 ms go at B=8: the M=8 small-GEMV band

Graph replay removes the launches, but the kernels still run. At B=8 the
families furthest from their own roofline per launch are the small-M weight
GEMVs — excess `(ms − bound) × count`, not launch dominated:

| kernel (face) at M=8 | launches | ms/launch | bound/launch | excess |
|---|---:|---:|---:|---:|
| up_proj (nvfp4, FFN) | 56 | 0.332 | 0.020 | 17.47 ms |
| out_proj (fp8, GDN) | 48 | 0.274 | 0.010 | 12.67 ms |
| down_proj (nvfp4, FFN) | 56 | 0.184 | 0.020 | 9.18 ms |
| gate_proj (nvfp4, FFN) | 56 | 0.167 | 0.020 | 8.23 ms |

These run at 3.5–12% of bound **per launch**, so the gap is the kernel, not
dispatch. The mechanism is the MMA band boundary: M≤8 decode GEMVs dispatch on
`linear_fp4_mma8` bf16, while M≥9 prefill uses the e4m3 WGMMA fp8 band
(`LINEAR_MMA_BANDS`). B=8 decode sits exactly on the wrong side of that
boundary — eight rows are still eight M=1-style mma8 GEMVs batched, not one
WGMMA GEMM. Batching to B=1 → B=8 moves bytes and graph time proportionally
(2.22×) but does not move the kernels to the fast MMA shape. Closing the 30%
number means serving decode through the M≥9 band (a fused batched GEMM, or
spec decode's wider verify tick), not a cheaper launch.

### Where the eager 47 ms go at B=1: launch overhead, ranked by `ms − bound`

Summing the B=1 bench table's per-row excess (eager `ms × count − roofline
bound`) gives **≈46.6 ms**, i.e. essentially the whole 53.2 − 6.6 gap. The top
three contributors are all small-M weight GEMVs whose own bound is near zero —
the cost is the launch, not the kernel:

| kernel (face) | launches | ms/launch | bound/launch | total excess |
|---|---:|---:|---:|---:|
| up_proj (nvfp4, FFN) | 56 | 0.135 | 0.020 | 6.44 ms |
| in_proj_b (nvfp4, GDN) | 48 | 0.111 | ~0 | 5.33 ms |
| in_proj_a (nvfp4, GDN) | 48 | 0.107 | ~0 | 5.14 ms |

(gate_proj nvfp4 follows at 5.04 ms, out_proj 4.61, down_proj 4.20.) Every
weight GEMV sits 0.09–0.14 ms per launch regardless of bytes; 497 launches ×
~0.094 ms ≈ 47 ms. Graph replay removes the launches from the tick — the single
remaining 1.8× gap to the 6.6 ms bound is kernel time plus graph pad/update
work, and is the actual remaining optimization target, not the 8× the bench
table suggested.

### Why the two clocks must be cited separately

- **`bench --kernels` `ms` column** = eager per-kernel timing, useful for
  ranking kernels by `ms − bound`; it is not served latency.
- **Engine-loop graph tick** = the served decode number.
The runbook §3 note now says this explicitly so the 12% row is not read as a
kernel-quality verdict.

## Rule

A per-kernel bench table over N launches measures dispatch overhead N times;
before attributing its total to kernel quality, time the same work through the
engine's actual dispatch path (CUDA graph replay = one launch). Rank kernels by
`ms − bound` only to find launch-bound populations — small-M GEMVs with
near-zero bounds — not to price the served tick.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | H20 node, physical card 6 (container `sglang-test`) | sm90 | B=1 graph 11.68 ms vs eager 47.99 ms (4.11×); B=8 graph 25.98 ms vs eager 64.69 ms (2.49×); roofline bounds 6.6 / 7.74 ms; 32-tick decode-only windows |

Raw artifacts: `/work/decode_clean_b8.json`, `/work/decclean6.log`,
`/work/roof6.log` (eager `bench --kernels` table), probe
`/work/probe_decode_clean_b8.py`.

Verbatim 32-tick medians (`/work/decode_clean_b8.json`, card 6):

```json
[
  {"b": 1, "graph": true,  "n_ticks": 32, "ms_min": 11.646, "ms_median": 11.684, "ms_p90": 12.102},
  {"b": 1, "graph": false, "n_ticks": 32, "ms_min": 47.520, "ms_median": 47.989, "ms_p90": 48.240},
  {"b": 8, "graph": true,  "n_ticks": 32, "ms_min": 25.794, "ms_median": 25.978, "ms_p90": 28.380},
  {"b": 8, "graph": false, "n_ticks": 32, "ms_min": 62.848, "ms_median": 64.689, "ms_p90": 71.163}
]
```
