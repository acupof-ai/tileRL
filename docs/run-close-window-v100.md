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
env → wait for `/health` → **assert the health body** → start the passive
reclaim sampler → run `probe_headroom_coldtail.py arm` → follower correctness
smoke → cancel-immediacy smoke → stop the serve. Artifacts land in
`$OUT/<arm>/` (`arm.json`, `steady.json`, `reclaim.json`, `follower.json`,
`cancel.log`, and a log per step).

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
TILERL_CLOSE_BUSYIDLE=1`.

`LIVENESS_POLL_S` is the load-bearing one: the supervisor's liveness probe sends
a **real chat every 60 s**, which lands inside the decode window being measured.
Set it back to 60 only in `--restore-only`.

## Arms

| arm | env delta | question |
|---|---|---|
| `baseline` | none | the reference every other arm is a delta against |
| `batch` | `TILERL_CLOSE_BATCH_D2H=1` | #741's one-sync close batch |
| `bg1` | `+ TILERL_CLOSE_BG_PUBLISH=1 TILERL_CLOSE_BG_DEPTH=512` | #743's background publisher at its own default depth |
| `bg2` | `+ TILERL_CLOSE_BG_PUBLISH=1` | the publisher with #745's **derived** depth (`(num_slots+1)·ceil(max_ctx/16)`) |
| `bg3` | `+ TILERL_CLOSE_BG_PUBLISH=1 TILERL_CLOSE_BG_DEPTH=8192` | the depth the 2026-09-20 window measured clean |
| `locksplit` | — | **refused**: #746 is not merged, and a flag nothing reads would report a no-op as a measured result |

`bg1` vs `bg2` is the depth question, not a second flag: `bg2` leaves the depth
unset so `build.py` derives it from the shape.

## Reading the result

Compare with the probe's own subcommand (it places the numbers, it does not
decide):

```sh
$PY scripts/probe_headroom_coldtail.py compare \
    --arms baseline=$OUT/baseline/arm.json bg2=$OUT/bg2/arm.json
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

The harness runs `scripts/steady_filter.py --log … --out $OUT/<arm>/steady.json`
on each arm for exactly this: it re-reads the same log under the standard set and
reports `steady_p50_ms` with the tail split out (`tail_n`, `tail_p50_ms`,
`tail_max_ms`). Each arm therefore carries both, and the one that matches the
sweep's口径 is `steady.json`.

Two properties worth knowing before trusting it:

- The tail split is an **absolute** 300 ms threshold, not a quantile. At the
  ~5-12 steady ticks a warm window yields, a 0.95 quantile cut separates nothing
  (measured: five ticks including one 5000 ms, and the cut landed on the maximum
  so the tail reported zero).
- A log written before the `path=`/`sparse=` tail fields existed cannot be shown
  steady. `steady_filter.py` reports those rows as
  `excluded_undecidable_n` with a note rather than counting them; the clause that
  rejects them is `sparse == 1`, not a separate absence check.

### Depth changes are two boots

`--depth` is written into the launch command, so a depth arm is a **separate
boot**, not something a single window process can switch mid-run. Two depths
means two full windows.

## What the window is trying to settle

- does `pub_cold_transfer` / `ssd_mmap` leave the close segment with steady
  decode held;
- `close_dev` vs `close_host` on the release ticks — device-busy or host-blocked
  (the #749 bracket);
- `ssd_mmap_worker` subtracted from `ssd_mmap` — how much of the step charge was
  cross-thread disk accounting;
- `reclaim.json` — the #740 trailing truncation, from apparent bytes only.

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
