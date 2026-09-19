# sm70 32k decode: ~10 tok/s is the steady physical wall; the per-request SSD close is the remaining design tail — 2026-09-19

> Status: measured, closed-loop. The steady 32k decode rate on one V100 sm70
> is a hardware wall, not a missed optimization; the one fixable subsegment in
> scope (#736 true draft query width) was landed and re-measured on device and
> gave the predicted small served gain. One defect remains open: the per-request
> cold SSD close (~3.1 s) and the shared spill file ignoring the 8 GiB cap.

## Context

ckl's question: can a single V100 sm70 serve Qwen3.8-27B-NVFP4 (f32 KV) 32k
decode near 50 tok/s, or is the observed rate a physical wall? Pre-registered
judgment: with spec depth 3, if effective tok/s is <20 **and** the steady model
segment is ≥120 ms, the steady rate is a hardware wall; if the model segment
collapses to tens of ms, it is a design problem to fix and re-measure.

One box, one process per arm, `scripts/probe_headroom_coldtail.py arm` against
an external hybrid serve: `--slots 4 --max-batch 4 --sparse-k 128
--sparse-min-tokens 8192 --scorer bounds --decode-graph --draft-attn-window-tokens
2048`, cold tier `--kv-cold-bytes 1GiB --cold-ssd-bytes 8GiB --cold-format f16`
(the current SSD-heavy form: 1 GiB pinned RAM, the rest on the #735 extent
spill). Each arm fills 2 independent 37.6k-token prompts (the word-stream prompt
tokenizes to 37,565–37,567 BPE ids, not 32,000 — identical for every arm) with 8
generated tokens, then 2 warm reps of 32 generated tokens; only the warm
decode ticks (dec=1) are scored. The fill guarantees a full tier before the warm
measurement. `TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0` logs every tick.

Arms: d1 at 2dc88a25; d3 at 2dc88a25; a tail d1 bracket at 2dc88a25 (drift
control); then d1 at d2c1d437 (#736) with `TILERL_DRAFT_TRUE_Q_WIDTH` unset
(the no-regression control) and =1. Same cold strategy, same prompt seeds.

## Measurements

Raw streamed tok/s undercounts spec output: accepted bonus tokens are coalesced
into chunks with the verified token, so chunks/s sees only the verify forwards.
Effective tok/s = `(decode ticks + accepted bonus) / warm first-to-last seconds`.

| arm (sha, flag) | raw tok/s (2 reps) | effective tok/s | accept rate | steady tick p50 | steady model p50 |
|---|---:|---:|---:|---:|---:|
| d1 @2dc88a25 | 5.17 / 5.04 | 9.43 / 9.49 | 0.82 / 0.88 | 183–187 ms | 168 ms |
| d3 @2dc88a25 | 4.56 / 4.35 | 10.71 / 10.75 | 0.49 / 0.45 | 201 ms | 169 ms |
| d1 bracket @2dc88a25 | 5.04 / 5.02 | 9.20 / 9.44 | 0.82 / 0.88 | 185–187 ms | — |
| d1 @d2c1d437 true-Q off | 5.07 / 5.07 | 9.24 / 9.55 | 0.82 / 0.88 | 183–184 ms | 167–168 ms |
| d1 @d2c1d437 true-Q on | 5.30 / 5.20 | 9.66 / 9.79 | 0.82 / 0.88 | 176–178 ms | 165 ms |

Bracket drift is −0.6% (5.044 → 5.015): the d1 numbers are stable. The true-Q
off arm matches the @2dc88a25 d1 baseline to the millisecond (model 167–168,
draft 12), so #736 introduces no regression and the on-arm delta is attributable
to the flag.

## Root cause

**Steady rate is a hardware wall.** d3 effective is 10.7 tok/s (<20) and the
steady model segment is 169 ms (≥120 ms) — both pre-registered arms hold. At a
steady decode tick the model trunk is 168–169 ms, ~90% of the ~185 ms tick;
sparse select is 1–2 ms and finalize is 11–18 ms on the positive-finalize
type-1 ticks (37–50 ms max), the d1 draft step is 12 ms. There is no
software segment left whose removal approaches 50 tok/s. d3 beats d1 on
*effective* tokens (10.7 vs 9.5) because each forward emits up to 4 tokens, even
though its raw streamed rate is lower and its per-verify cost is higher (24 ms
draft, lower accept 0.45–0.49).

**The one design-fixable steady subsegment was the draft query bucket.** A d1
decode-only draft verify has true query width q = n_ok+1 = 1–2, but the head
rounded it up to the 64-wide prefill bucket, multiplying attention/projection
work for rows no kernel reads (#736). With the flag on, the steady `draft_step`
segment drops 12 → 7 ms (−42%), the steady tick drops 183 → 177 ms, and served
effective tok/s rises 9.4 → 9.7 (**+~3%**), with acceptance byte-identical
(14/15 accepted both arms). The win is real but bounded by the 168 ms trunk —
exactly why #736 shipped opt-in rather than a default flip.

**The long tail is a separate, per-request design cost, not steady decode.**
Once per request, at close, the engine transfers cold shared pages to SSD:
`release_close_request` ≈ 2.95–3.18 s, inside it `ssd_mmap` 1.81–2.03 s and
`pub_cold_transfer` 1.71–1.87 s. Long decode ticks (>300 ms) are 5.9% of warm
decode ticks on every d1 arm and 15.4%/7.1% on the two d3 reps; they do not
move the steady median but they dominate worst-case request latency. This is
the SSD-heavy 1 GiB-RAM form; the 2026-09-17 full-tier finalize numbers
(190–222 / 729–1138 ms, 8 GiB RAM tier) are a different tier shape and are not
comparable — the warm steady `sparse_finalize` segment here is a 11–18 ms
median (37–50 ms max across the ten reps), and the tail moved from finalize to
the request-close SSD transfer.

A separate fill-phase outlier is raw-log-only, not in the vendored warm JSON:
in the d3 serve log `serve-d3.boot`, tick 198 is a 9978 ms decode tick
(`model=9952ms`, `offers_pages=0`, `ssd_mmap=0`) immediately after two
7446/7914 ms prefill ticks (196/197) during a cold fill — a hollow model
forward at the fill edge, distinct from the steady trunk and from the SSD
close tail. Tick 199 (6625 ms) is the same shape.

## Fix

- #736 (true draft query width), already merged at d2c1d437, is device-verified
  here: −42% draft step, +~3% served effective d1, no acceptance change, no
  regression vs the off control. Stays opt-in.
- `scripts/probe_headroom_coldtail.py` cold-fullness gate was RAM-only
  (`kv_cold_bytes + kv_cold_shared_bytes`) and could never pass on a 1 GiB RAM
  tier, so with warm reps it refilled until the high-water spill file filled the
  disk. It now sums all four keys (private/shared × RAM/SSD) and fails fast on
  the first rep whose gate fails. Branch `probe/cold-gate-ssd-2026-09-19`.
- Open, named not landed: (1) amortize or defer the per-request cold SSD close
  (~3.1 s ssd_mmap + pub_cold_transfer) so it is not on the close path; (2) the
  shared prefix spill file ignores `--cold-ssd-bytes` — it reached 13.6 GiB
  logical (25 GiB physical high-water) against an 8 GiB cap, because the shared
  prefix cache publishes pages that nothing serves and the extent file never
  shrinks. Tracked in OPEN.md.

## Rule

Judge decode throughput on effective tokens `(forwards + accepted bonus)`, not
raw chunks/s — spec coalesces bonus tokens into chunks and undercounts depth.
Separate the steady median from the per-request tail before calling a rate a
wall or fixable: here the 168 ms trunk (90% of the tick) is hardware while the
3.1 s SSD close is design, and the fixable 12 ms draft segment was worth +3%
served, not the +50% target. Always run the merged-but-off arm as the regression
control before the on-arm delta; matching the prior baseline to the millisecond
is what attributes the on-arm gain.

## Provenance

Vendored arm summaries (script JSON, 5 arms):
`v100-32k-verdict-2026-09-19/{d1-r3,d3-r1,d1-bracket,tqoff-r1,tqon-r1}.json`.
Raw per-tick segment timing is in the V100 logs `~/tilerl-logs/serve-d1.boot`,
`serve-d3.boot`, `serve-d1-bracket.boot`, `serve-tqoff.boot`, `serve-tqon.boot`.
Trees 2dc88a25 (baselines) and d2c1d437 (#736), verified on box via
`.synced_commit` before each arm.
