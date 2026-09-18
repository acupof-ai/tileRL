# Draft read window W=2048: per-forward win is real, end-to-end not significant — 2026-09-19

> Status: measured, decision = **do NOT flip the default**. The draft decode
> READ window stays at full prefix (W=0) as the default; 2048 remains the
> opt-in `DRAFT_ATTN_WINDOW_TOKENS_RECOMMENDED` (#723, default 0). The
> 2026-09-17 modeled +53% @32k was explicitly not proof; this n=30 device run
> is the sign-off, and it does not clear the bar.

## Context

Second-half device measurement in the same V100 sm70 stop window as the
headroom arms. Qwen3.8-27B-NVFP4 + MTP d1, `--sparse-k 128`, spec d1, in-process
eager probe `scripts/probe_draft_window_sweep.py`, real wikitext-103 **train**
text (a bounded non-empty prefix; see Corpus below), NoPrefixStore, direct
submit (think-off), 96 generated tokens, **n=30 paired disjoint spans per
length**, the same span list reused across every W arm. Bare engine, GPU
exclusive, no serve/liveness. Main run W=0/1024/2048 at 32768 and 16384; a
separate tail **W=0 bracket** re-ran W=0 at both lengths to measure run drift.

Go/no-go bar (pre-registered): flip only if W=2048, after subtracting the
first-vs-tail W=0 bracket drift, still shows **≥+20% served/measured tok/s**,
the W=0 bracket closes, `accept_len` drops ≤3% and `accept_rate` ≤0.04, and no
arm is inert (`proof=ok`).

## 1. Mechanism holds — the decode-region draft step is 6–10× cheaper

The window engages correctly (`proof=ok` every W>0 arm; `windowed_seq_len`
tracks 1024/2048). Segmenting the per-forward `[draft-timing] gpu ms` by
context position (the JSON stores only a mixed median, so this is from the raw
log, bucketed by arm and `max_seq`) separates the 96-token decode region from
the per-prompt cold fill:

| length | decode region (high pos) W=0 | W=1024 | W=2048 |
|---|---:|---:|---:|
| 32768 (≥24k) | 118 ms | 10.1 ms | 12.4 ms |
| 16384 (≥12k) | 59 ms | 9.9 ms | 12.2 ms |

The cold-fill/low-position draft forward (~138 ms mid, up to ~435 ms during the
32k fill) is identical across all three W — the window trims the trailing read
of the steady decode step, not the fill. Acceptance cost passes the
**pre-registered gate of accept_len drop ≤3% and accept_rate drop ≤0.04** (the
tighter ~1.5%/~0.025 figures seen here are an observation, not the threshold),
at full JSON precision:

| length | accept_rate Δ (W=2048 vs 0) | accept_len Δ | vs gate |
|---|---:|---:|---|
| 32768 | −0.0246 (0.7571 vs 0.7817) | −1.483% (1.7486 vs 1.7749) | inside 3% / 0.04 |
| 16384 | −0.0256 (0.7153 vs 0.7410) | −1.514% (1.7075 vs 1.7338) | inside 3% / 0.04 |

## 2. End-to-end: no-go — gains are below +20% and the 16k bracket does not close

Per-W `tok_s_med` (n=30) with both W=0 baselines:

| length | W=0 first | W=0 tail | bracket Δ0 | W=2048 | vs first | vs tail (conservative) |
|---|---:|---:|---:|---:|---:|---:|
| 32768 | 6.17 | 6.01 | **−2.6% (closed)** | 6.95 | +12.6% | **+15.6%** |
| 16384 | 4.86 | 7.54 | **+55.1% (NOT closed)** | 8.32 | +71.2% | **+10.4%** |

W=1024 is worse than both 0 and 2048 (32k −24.8% vs tail W=0; 16k −21.2%), so
1024 is not a candidate either.

- At **32k** the bracket is tight (Δ0 −2.6%), but the conservative W=2048 gain
  is only **+15.6% < 20%** and the curve is non-monotonic (1024 below 0).
- At **16k** the bracket **fails**: the tail W=0 run is 55% faster than the
  first W=0 run (7.54 vs 4.86), a same-config drift larger than the window
  effect. The headline "+71%" is an artifact of a slow first run; against the
  tail baseline W=2048 is only **+10.4%**. The bracket — not the median alone —
  is what prevents a false positive here.

## 3. Why the per-forward win does not show up end-to-end

Two fixed/system terms dominate the decode wall time and are independent of W:
- **Cold fill** is a per-prompt fixed cost that does not change with the read
  window: median 156 s (32k) / 63 s (16k), identical across the W arms.
- **Trunk + per-tick finalize/lock jitter** at sm70 sparse d1 is the long-tail
  documented in `2026-09-17-cold-tier-full-finalize-relocation-long-tail.md`
  (~150 ms launch floor plus 7–8× finalize batches and hollow-forward ticks at
  the card edge). A 100 ms saving on the draft step is within the run-to-run
  swing of those terms.

The draft window removes a real ~100 ms/step at 32k, but that is smaller than
the system noise measured between two identical W=0 runs.

## 4. Decision

- Default stays **W=0 (full prefix)**. Do not flip to 2048.
- 2048 stays the opt-in recommendation (`#723`); a workload that wants it can
  set `--draft-attn-window-tokens 2048` and gets the cheaper draft step with
  ~1.5% lower accept_len.
- W=1024 is not used: it was slower end-to-end than both 0 and 2048.

## 5. Re-test preconditions

Re-measure W only after the system terms that mask it are reduced:
1. Land the cold-tier O(1) finalize / per-tick lock fix (the root-cause work
   linked from the 2026-09-17/18 cold-tier and long-step-tick entries) so the
   100 ms draft saving is not inside the finalize/jitter band.
2. Run an instrumented sweep that writes **per-prompt** tok_s (IQR / p10–p90),
   not only the cross-30 median. This run's JSON keeps `tok_s_med` only; the
   bracket Δ0 was enough to reject the flip, but a positive claim needs the
   within-arm distribution to show the gain clears the noise.

## Corpus and provenance

- Real wikitext-103 **train** text. The full train split (~540M chars, two
  parquet shards) OOMs the 31 GiB host when tokenized at once; the run used a
  bounded non-empty train prefix (~1.02M tokens, empty/header rows skipped),
  supplied via `--corpus-glob`. The n≥30 hard gate passed (n_eff=30/30 at both
  lengths), id 0 count in the consumed prefix was 0, and spans were paired
  across W. The proper fix landed as #733
  (`wikitext_ids_stream(max_tokens=…)` char-budget bounded encode + split-glob
  expanduser); future runs use `--split train` directly.
- Data vendored alongside this entry (bench evidence is not hand-transcribed):
  `main-wsweep.json` (the 6 W×length aggregates, script JSON) and
  `w0-bracket.json` (tail W=0 at both lengths). Raw per-forward timing is in
  the V100 run log `~/tilerl-logs/wsweep.log` (main) / `wbracket.log`.
- Tree 39b6f100 (probe/corpus pre-#733; corpus read used the bounded parquet);
  probe run via `scripts/probe_draft_window_sweep.py --sparse-k 128 --time-draft
  --split train --min-prompts-per-length 30 --lengths 32768,16384 --windows
  0,1024,2048 --prompts 30`.

## Rule

A ~10× per-forward kernel win does not justify a default flip when the saved
term is smaller than same-config run-to-run drift. Always bracket a sweep with
a repeated baseline arm and require the treatment to beat the *worst* baseline
by the pre-registered margin; a median above one baseline run is not a result
when an identical rerun moves more than the treatment.
