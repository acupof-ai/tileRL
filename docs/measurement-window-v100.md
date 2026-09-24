# V100 measurement window — how to run one

The procedure for a single-sitting measurement window on the V100 (sm70). This
page is **how to run**, not what was found: results live in
`docs/experience/{wins,errors}/`, and the flags they measured are described
there. The one thing this page owns is the order and the discipline that make a
window's numbers comparable to the ones already recorded.

Read with [serve-v100.md](serve-v100.md) (the launcher and its env knobs) and
[PENDING-REMOTE-CARDS.md](experience/PENDING-REMOTE-CARDS.md) (which gates are
still open). The claim rules this page enforces are the ones in the agent
contract: a mechanism claim ships with the probe that tested it, an arm's
number needs a placement control, and a gate is green only after its negative
control is red.

## 0. Before the window

**Freeze what will run.** Every number is attributed to a tree, so the serve log
must carry the sha it booted. `.synced_commit` in the serve tree is the
authority — read it on the box before the first arm, and do not trust a tar
timestamp (a pending tar newer than the running process is not a newer process).

**Name the card owner.** The claim table says who is *using* a card, not whose
it is. An idle card with no claim still reads as available; ask before taking
one nobody told you was yours.

**Pre-register the arms and the bar.** Write down, before any run, the arm list,
the control, and what result would be a reject. The recorded windows all did
this and it is why their negative verdicts are readable — e.g. the 32k decode
verdict's pre-registered condition ("effective tok/s < 20 **and** steady model
segment ≥ 120 ms ⇒ hardware wall") is what made its wall conclusion a
measurement rather than an opinion
([errors/2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md](experience/errors/2026-09-19-sm70-32k-decode-physical-wall-and-ssd-close-tail.md)).

**Check the probe's API against the deployed tree, not against a laptop copy.**
`py_compile` does not catch a wrong-arity call. Import the probe against the
tree that will run it and inspect the signature before the window opens — a
signature mismatch discovered mid-window costs the downtime, not the diff.

## 1. Boot the serve for an arm

Each arm is the SAME tree booted with different **environment**, never a
different code path. The hybrid launcher is
[`scripts/serve_hybrid_v100.sh`](../scripts/serve_hybrid_v100.sh); every knob in
it is env-overridable and the flag set it passes is fixed, so an arm that needs
a different flag value sets the env the launcher reads rather than editing the
script.

Two env rules decide whether an arm is even the arm you think it is:

- **`TILERL_DRAFT_TRUE_Q_WIDTH` and `TILERL_DRAFT_ATTN_WINDOW_TOKENS` are read
  at import / build time.** They cannot be flipped mid-process. A "W on then W
  off in one serve" arm is not possible; those are two boots.
- **`TILERL_STEP_TIMING=1` and `TILERL_STEP_TIMING_SLOW_MS=0`** for any window
  that reads per-tick segments. `SLOW_MS=0` logs **every** tick; the default
  500 ms logs only the slow tail, which is not a distribution and cannot answer
  a percentile question.

**Silence the liveness watchdog before measuring.** `serve_liveness.py` is not
only a `/health` poll: while `slots_used < slots_total` its slot-leak fallback
issues a real chat completion (`max_tokens: 4`) once per `POLL_S`, which lands
inside the decode window and is indistinguishable from served traffic in the
tick log. Set `LIVENESS_POLL_S=999999` for the measurement, and **restore 60 s
when the window ends** — that fallback is the idle self-heal, not something to
leave off.

Per-arm boot gate, before any measurement:

1. `/health` 200, **current-boot** warmup done (a >15 s first call is compile
   warm-up, not a result — retest after it);
2. the boot's first log line sha equals the intended short sha;
3. the arm's env is actually set. The launcher's argv never carries these, so
   read the live process: `tr '\0' '\n' < /proc/<pid>/environ | grep TILERL_`.

## 2. Arm order, and why the control goes both ends

Run the **control first and last**. A same-tree re-run at the end is the drift
control: if the first and last control arms do not agree within the quantity's
own noise, nothing measured between them is attributable. The recorded W=2048
sign-off is the case that proves it — at 32k its bracket closed (Δ0 −2.6%) and
the conservative gain was +15.6%, below the pre-registered +20%; at **16k the
bracket failed outright**, the tail W=0 run measuring **55% faster than the
first** (7.54 vs 4.86 tok/s), a same-config drift larger than the effect being
measured, which turned a headline "+71%" into +10.4%. The bracket, not the
median alone, is what prevents that false positive
([errors/2026-09-19-w2048-window-end-to-end-not-significant.md](experience/errors/2026-09-19-w2048-window-end-to-end-not-significant.md)).

Also run the **merged-but-off arm** before any on-arm: a flag's gain is the
on-arm minus the off-arm *on the same tree*, and an off-arm that matches the
prior baseline to the millisecond is what attributes the delta to the flag
rather than to the tree.

Cold tier must be the same shape in every arm, and the **fill must complete
before the warm measurement**. Fill and warm are separate phases; only warm
decode ticks are scored.

## 3. What a window can and cannot measure

| question | instrument | note |
|---|---|---|
| steady decode rate + per-tick distribution | `probe_headroom_coldtail.py arm` | one arm per boot; `--warm-reps` (default 3) refills and re-warms per rep, so the rate is a median with a spread |
| per-prompt tok/s spread (IQR / p10–p90) | `probe_draft_window_sweep.py` | in-process, one engine; W mutated live |
| d1 vs d3 | **not in the sweep probe** | `probe_draft_window_sweep.py` hardcodes `spec_depth=1` and has no `--depth`; the launcher hardcodes `--depth 1`. A depth arm is a **separate launcher env change** (or `scripts/ab_draft_depth.py`, which exists for the depth question specifically) |
| true-Q on/off | boot env `TILERL_DRAFT_TRUE_Q_WIDTH` | import-time, so two boots |
| cap reclaim over time | `probe_headroom_coldtail.py reclaim-sample` | passive sampler; drives no requests |
| W×R over many prompts, one arm per process | `probe_wr_sweep_worker.py` + `wr_sweep_report.py` | the cross-process half of the sweep: each arm is a fresh engine, so a W that mutates import-time state is not carried between arms. `probe_wr_sweep_worker.py` drives one arm and writes its rows; `wr_sweep_report.py` reads a directory of them into the comparison table. Run both by hand on the box (see the worked invocation in the sparse-WR entry) |

The W×acceptance sweep is paired: the same prompts run in every W arm within a
length, so a between-W difference cannot be a between-passage difference. Keep
that property — re-drawing prompts per arm voids the comparison.

Two harnesses implement it and the choice is about process boundary, not
features: `probe_draft_window_sweep.py` sweeps W **in-process** (one engine, W
mutated live) and is the cheaper one when nothing about W is import-time;
`probe_wr_sweep_worker.py` + `wr_sweep_report.py` give each arm **its own
process**, which is what a W that has to be set before the engine builds
requires. Both keep the pairing rule above.

## 4. Reading a tick

Score **only** the steady decode set. The serve-line filter is
`dec=1 & sparse=1 & model>0 & sample>0` with `path != graph`; it drops idle and
untimed ticks.

**The two probes do not apply that filter identically today.** Say which one a
number came from:

| probe | what it keeps | consequence |
|---|---|---|
| `probe_draft_window_sweep.py` | the full steady set above, with the closing tick split out | per-prompt band is a steady tick band |
| `probe_headroom_coldtail.py` | `dec > 0` only (its `is_decode`), plus its own type1/type2 split | a graph/idle decode-tagged tick is not excluded, and the closing tick is not separated |

So a p50 from the headroom probe and a p50 from the sweep probe are the same
statistic over **different sets**. Do not put the two columns in one table as if
they were interchangeable.

**The way across is `scripts/steady_filter.py`**, which re-reads a log under the
standard set so a headroom arm and a sweep arm can be placed side by side — or
not placed at all when the log cannot support it. It is where that set is
defined, as the symbol `STANDARD_FILTER` and the predicate `is_standard`;
`summarise` reports the steady median with the long close-tail ticks split out
via `tail_ms`, and `parse_rows(log_path, offset, until)` windows the read by byte
span. Two conventions live in it deliberately and must not be mixed: `median` is
the true median (`statistics.median`, averaging the middle pair on even n) while
`pct` is nearest-rank — a tick duration is quoted on the former, a percentile on
the latter. Prefer these symbols over re-deriving the set, and cite them by name
rather than by file so a later move does not silently break the reference.

**The closing tick is not a steady tick.** A request's last tick has the model
at its steady cost and `sample` taking over the whole tick (e.g.
`dec=1 model=165ms sample=5440ms`), once per request at the dec→prefill
boundary. Where the instrument separates it, it is counted **separately**, never
folded into the band — it is the single largest way a distribution gets silently
flattered at the tail.

**Report a median and a spread, not a raw rate.** Raw streamed tok/s moves with
whether a closing tick landed inside the measured span. Steady state is the tick
median (p50/p90) plus the per-prompt band; raw is a whole-run feel number and
must say whether the span contained a close.

**Effective tok/s ≠ raw tok/s under speculation.** Accepted bonus tokens coalesce
into the same chunk as the verified token, so a chunk counter sees only the
verify forwards. `effective = (decode ticks + accepted) / warm decode seconds`.

## 5. Cold tier: the gate is four keys, not one

A RAM-only cold check can never pass on the SSD-heavy shape. The occupancy that
matters is the sum over **both tiers × both pools**:
`kv_cold_private_bytes + kv_cold_shared_bytes + kv_cold_private_ssd_bytes +
kv_cold_shared_ssd_bytes` — the probe's `_cold_occupancy`. Gate on that total
and print the four components, so a fill that looks full because it is all on
one side is visible.

**Report the fill state with any tok/s number.** Full-tier and empty-tier serve
rates differ by 1.4–2.95x on this line, so a rate without its cold-tier state is
not comparable to anything
([errors/2026-09-17-cold-tier-full-finalize-relocation-long-tail.md](experience/errors/2026-09-17-cold-tier-full-finalize-relocation-long-tail.md)).

## 6. Stopping between arms

TERM the supervisor pid (the parent of the `tilerl.cli serve` child), sleep ~11 s,
then confirm for supervisor / serve / guard with `ps -o stat= -p <pid>` that
each is **gone**. A zombie counts as alive to `kill -0`, `/proc/<pid>` and
`pgrep -f` alike, so those are not the check. Require
`nvidia-smi --query-gpu=memory.used` = 0 MiB and an empty
`--query-compute-apps` before the next boot.

**Do not restart a wedged serve just because it looks idle.** A live process is
evidence; if `/health` answers but `decode_forwards` is frozen, that is the
finding. Preserve the log and report.

## 7. Ending the window

- **Restore the box**: `LIVENESS_POLL_S` back to 60 s, measurement env gates
  unset, then verify with a real `/v1/chat/completions` `max_tokens=8` that
  returns 200 with `finish_reason=stop`. The recorded windows end with exactly
  this smoke test.
- **Vendor the evidence.** Every number in the doc must point at an artifact in
  the tree: the probe's JSON under
  `docs/experience/{wins,errors}/<slug>-<date>/`, plus the raw serve logs named
  by path. Hand-transcribed numbers are not evidence. Findings with no rerun
  path must say so explicitly rather than being presented as reproducible.
- **Land the entry the same day** (wins or errors), and a CHANGELOG line if the
  window produced a phase exit, a default flip, or an accept-or-reject verdict.
- **An entry that names a fix is not a fixed bug.** Anything whose fix has not
  landed says `Status: open` and is listed in `docs/experience/OPEN.md`.

## Rule

A window's numbers are comparable only when the tree is named, the control sits
at both ends, the cold tier is the same and reported, the closing tick is split
out, and every figure traces to a vendored artifact. The order above is what
makes that true; skipping a step does not make the window shorter, it makes its
result unattributable.
