# V100 close-tail window: one-key harness and recovery checklist

`scripts/run_close_window_v100.sh` runs the final close-tail window in one order
and writes every artifact to a vendored directory. This page is what to do when
it is not the script's job to decide: recovering the box, and reading the result.

## Running a window

```sh
scripts/run_close_window_v100.sh --list                 # arms, no serve started
scripts/run_close_window_v100.sh --arm baseline         # one arm
scripts/run_close_window_v100.sh --all                  # every arm, then prompts to restore
```

Per arm, in order: stop any serve → boot `serve_hybrid_v100.sh` under that arm's
env **with its own `SERVE_LOG`** → wait for `/health` → **assert the health body**
→ run `probe_headroom_coldtail.py arm` → re-window
the steady filter → follower correctness smoke → cancel-immediacy smoke → stop the
serve. Artifacts land in `$OUT/<arm>/` (`serve.log`, `arm.json`, `steady.json`,
`reclaim.json`, `follower.json`, `cancel.log`, and a log per step).

A failed smoke fails the arm, and the exit code says which:

| rc | meaning |
|---:|---|
| probe's own | passed through (13 = fail-closed on too few good reps) |
| 3 | follower **MISMATCH** — the store answered with the wrong tokens |
| 4 | follower **NO-PREFIX-HIT** — same tokens, no hit; the store did not serve |
| 5 | follower **block leak** — `blocks_used > blocks_total` |
| 6 | cancel smoke failed |

MISMATCH and NO-PREFIX-HIT are different findings with different responses — a
mismatch is a correctness bug, a missing hit is a store that did not serve — so
they do not share a code.

### The health gate

An answering `/health` is not the right server. A leftover tiny-model process, a
different pool size, or a runtime decode-graph fallback all answer 200, and the
2026-09-19 window lost time to each. The gate asserts `model=qwen38-27b`,
`blocks_total=2213`, and `decode_graph` not `false`, and refuses to send a
request otherwise. Override with `EXPECT_MODEL` / `EXPECT_BLOCKS` only when the
window genuinely runs a different shape.

### Instrumentation every arm gets

`LIVENESS_POLL_S=999999 TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0
TILERL_DRAFT_ATTN_WINDOW_TOKENS=2048`.

`LIVENESS_POLL_S` is the load-bearing one: the supervisor's liveness probe sends
a **real chat every 60 s**, which lands inside the decode window being measured.
Set it back to 60 only in `--restore-only`.

`TILERL_DRAFT_ATTN_WINDOW_TOKENS` is instrumentation, not a treatment: the probe
asserts the window (`--expect-window 2048`) and refuses the arm (rc 13) when it
does not match, and the loader default is `W=0`. The shipped serve passes no such
flag, so this env is injected per arm by the harness and never by the launcher.

## Arms

| arm | env delta | question |
|---|---|---|
| `baseline` | none | the reference a measured arm would be a delta against |

`baseline` is the only arm left. The `batch`/`bg1`/`bg2`/`bg3`/`bgcap` arms and the
`locksplit` placeholder were deleted with the machinery they measured (#784, #787):
`TILERL_CLOSE_BATCH_D2H`, `TILERL_CLOSE_BG_PUBLISH` and `TILERL_CLOSE_BG_DEPTH` are
no longer read in `src/`, so an arm setting them would have silently measured
`baseline` and reported it as a treatment, and `locksplit` refused to run at all
(#746 never merged). `TILERL_COLD_PREFIX_SSD_CAP` survives in `src/` as a cold-tier
capacity knob with no arm here; it is verified per-M6 rather than through a
measurement arm.

The harness still runs the full per-arm pipeline (boot, health gate, probe, steady
re-filter, follower and cancel smokes) against `baseline`, and
`--clean-spill-only` deletes the regenerable spill with no serve at all.

## Reading the result

Compare with the probe's own subcommand (it places the numbers, it does not
decide):

```sh
$PY scripts/probe_headroom_coldtail.py compare \
    --arms baseline=$OUT/baseline/arm.json
```

### Two steady-tick filters — do not put them in one table

`probe_headroom_coldtail.py` keeps ticks on **`dec > 0`**, a wide set that
includes captured-graph ticks and the long close-tail ticks. The sweep arms are
quoted on the tighter standard set:

```
dec == 1 and sparse == 1 and model > 0 and sample > 0 and path != graph
```

with ticks over 300 ms listed as tail rather than folded into the median. Those
are **different statistics**, so a headroom arm's `p50_ms` and a sweep arm's p50
must not sit in the same before/after table.

The harness runs `scripts/steady_filter.py --log $OUT/<arm>/serve.log --window …`
on each arm for exactly this: it re-reads the same log under the standard set and
reports `steady_p50_ms` with the tail split out (`tail_n`, `tail_p50_ms`,
`tail_max_ms`). Each arm therefore carries both, and the one that matches the
sweep's口径 is `steady.json`.

Three properties worth knowing before trusting it:

- The tail split is an **absolute** 300 ms threshold, not a quantile. At the
  ~5-12 steady ticks a warm window yields, a 0.95 quantile cut separates nothing
  (measured: five ticks including one 5000 ms, and the cut landed on the maximum
  so the tail reported zero).
- A log written before the `path=`/`sparse=` tail fields existed cannot be shown
  steady. `steady_filter.py` reports those rows as
  `excluded_undecidable_n` with a note rather than counting them; the clause that
  rejects them is `sparse == 1`, not a separate absence check.
- **The read is windowed to the arm's warm spans, and BOTH ends matter.** Standard
  set is not the same as steady state: the supervisor's warmup (dense 7000 +
  sparse 9000, 8-token decodes) and each rep's cold **fill** write short-context
  decode ticks that pass the standard set — the fill's land in the tail, the
  warmup's in the median.

  A span needs the rep's **own** end, not the next rep's start. `log_byte_offset`
  is taken *after* the fill and *before* the warm POST, so it is a warm **start**:
  `[off_i, off_{i+1})` contains `warm_i` **and** `fill_{i+1}` — the very ticks this
  is meant to drop, and only the last rep would be clean that way (and only
  because nothing happens to write after it). So `arm.json` carries
  `log_byte_end` per rep (taken when the warm returns, before the next fill
  exists) and the harness passes one `--window start:end` **per rep, the last
  included**, in a single call. Averaging per-rep medians instead would weight a
  2-tick rep the same as an 8-tick one.

  `probe_headroom_coldtail.py` reads its per-rep `p50/p90/max_ms` and
  `frac_over_300` through the same window, so its numbers move with this too: a
  rep followed by a fill was reporting that fill's tick in `max_ms` (measured
  176 ms warm-only against 400 ms).

  Consequence for reading `steady.json` at all: `windows` records the spans the
  median is over. `[[0, null]]` means the whole file was read, so on a close-window
  arm that number includes the supervisor's warmup and is not the steady figure.

### Depth changes are two boots

`--depth` is written into the launch command, so a depth arm is a **separate
boot**, not something a single window process can switch mid-run. Two depths
means two full windows.

## What the window is trying to settle

- does `pub_cold_transfer` / `ssd_mmap` leave the close segment with steady
  decode held;
- `reclaim.json` — the #740 trailing truncation, from apparent bytes only (only
  if the sampler was run by hand; this harness no longer starts it).

### The reclaim sampler

**Not started by this harness.** It was wired to the `bgcap`/`bg2` pair, and those
arms are gone with the close/batch/bg machinery (#784). The probe's `reclaim-sample`
subcommand is untouched: run it by hand against the shared prefix spill
(`<cold-ssd-path>.prefix.bin`) when a window is about the #740 trailing truncation.

Two things set its span, both arithmetic on measured numbers, and both still apply:

- **It must still be running when rep0's first release happens.** One 32k cold
  fill prompt costs ~156 s measured; the probe's `--fill-n` default is **5** (the
  harness does not pass it) and the warm request is itself a 32k prompt, so the
  first release is `(5+1)·156 ≈ 936 s` in. A `60 × 10` span (`590 s`) ends ~350 s
  *before* the event it exists to sample — the rows would show only the plateau.
- **It must not run far past the probe.** At this shape the probe is the longer of
  the two (3 reps × 936 s ≈ 47 min), so the sampler's span fits inside the arm
  rather than setting it.

```sh
scripts/probe_headroom_coldtail.py reclaim-sample \
    --spill-path "$ROOT/sparse_cold_128k.prefix.bin" --out "$OUT/reclaim.json" \
    --samples 90 --interval-s 15
```

## Recovery checklist

The window leaves the box on an experimental arm. Getting back to the shipped
serve:

1. `scripts/run_close_window_v100.sh --restore-only` — stops the arm's serve,
   boots `serve_hybrid_v100.sh` with **no** experimental flags and
   `LIVENESS_POLL_S=60`, re-runs the health gate, and sends one short chat.
2. **Confirm the serve is really down before touching the spill.** The script
   refuses to delete while `tilerl.cli serve` is running, and asks per file:
   ```sh
   pgrep -f "tilerl.cli serve"     # must be empty
   pgrep -f "serve_liveness.py"    # must be empty
   ps -o stat= -p <pid>            # Z = zombie; kill -0 and pgrep call it alive
   ```
3. Delete the regenerable spill files only after that, and only if the next
   window can refill them: `sparse_cold_128k.bin` and its `.prefix.bin`. The
   script's interactive cleanup does this; there is no `--force`.
4. `--restore-only` is also the answer after a failed arm: it does not need the
   arm to have succeeded.

### If a boot never becomes ready

The supervisor's own fuse is the first thing to read
(`$ROOT/.servehybridsse.fuse` and the `servehybrid: ...` lines in
`$ROOT/servehybridsse.log`); it exits 2 on a crash burst rather than looping.
`MAX_RESTARTS`, `RESTART_FUSE_MAX` and `RESTART_FUSE_WINDOW_S` are the
supervisor's, not this harness's.

## Rule

A window harness exists to make the manual steps that already cost time
impossible, not to measure: assert the shape you are comparing under, turn the
instrumentation on the same way every arm, and make recovery a single command.
