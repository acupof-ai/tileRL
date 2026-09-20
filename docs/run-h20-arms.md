# H20 (sm90) 32k arm matrix: runbook

The H20 measurement matrix for the 32k decode arm: six serve arms, one card, one
arm per boot. This page is the arm list, the serve env each arm pins, and the
reading rules — **commands and parameter names, not a scheduler.** The arms are
launched by hand through `scripts/pod_run.sh` (see [serve-h20.md](serve-h20.md));
nothing here automates them.

Read with [measurement-window-v100.md](measurement-window-v100.md) (the window
discipline this mirrors) and [run-close-window-v100.md](run-close-window-v100.md)
(its one-key harness). The claim rules are the agent contract's: a mechanism
claim ships with the probe that tested it, an arm's number needs a placement
control, and a gate is green only after its negative control is red.

The口径 below (steady set, the two rates, the graph verdict, the cold gate) is the
Team A consensus recorded in the org board's 当前事实 section (2026-09-20), not a
second convention — this page states it inline so it stands alone, and the board
is where a conflict is settled.

## Protocol every arm shares

- **One arm per boot.** A different depth, k, graph mode or draft window is a
  different *boot*, never a flag flipped mid-process: the serve CLI resolves them
  at build time (`--depth` lands in the launch command; the draft window is read
  at load). Two values of one key means two full boots.
- **Silence the liveness watchdog before measuring.** `serve_liveness.py` sends a
  real chat completion (`max_tokens: 4`) once per `POLL_S` while a slot is free,
  which lands in the decode window and is indistinguishable from served traffic.
  Set `LIVENESS_POLL_S=999999` for every arm; restore the 60 s default when the
  matrix ends. On H20 `slots=8` makes that injected request certain, not
  occasional.
- **Certify the boot from `/health`, never from the boot line.** The launcher's
  `arm:` banner is an env descriptor, and an env value is not the resolved one —
  see "The graph-off arm" below.
- **The serve log is the tick artifact.** One boot appends its own `serve_h20:
  tree <dir> sha <10> boot N <arm> at <ts>` line and its `[step-timing] tick …`
  lines to the same `/work/serve_h20.log`, so an arm is the line span between its
  boot line and the next. Cite that span.

## Reading one arm

Two rates, both reported, from two instruments. Neither is derived from the
other.

**Tick level — `scripts/steady_filter.py` (the only implementation).** It owns
the standard steady set and the two percentile conventions; do not re-write the
filter here or a second copy drifts from it (that is why it exists). It reports
the steady median with the long close tail split out.

```sh
python3 scripts/steady_filter.py --log /work/serve_h20.log --out <arm>/steady.json
```

The standard set it applies is `STANDARD_FILTER` (`dec==1 and sparse==1 and
model>0 and sample>0 and path!=graph`), and the tail split is an absolute
threshold, not a quantile — at the ticks a warm window yields, a quantile cut
separates nothing. A tick-duration median is `statistics.median` (the true median
that averages the middle pair on even n), **not** nearest-rank; `pNN` is
nearest-rank. Do not cross the two.

**A5 is the one arm the standard set cannot reach.** `is_standard` already pins
`sparse == 1`, so the dense arm's ticks are excluded by definition and the
intersection is empty. A5 is read with `probe_h20_arm_read.py log --arm-sparse 0`,
which drops that one clause and pins `sparse == 0` itself
(`dec==1 and sparse==0 and model>0 and sample>0 and path!=graph`); reading it with
the sparse filter reports zero ticks from a log full of them.

```sh
python3 scripts/probe_h20_arm_read.py log /work/serve_h20.log --arm-sparse 0 \
    --from-line <this boot's line> --to-line <next boot's line - 1>
```

The line window is part of the command, not an option: without
`--from-line`/`--to-line` the reader covers the whole file and reports every
arm's ticks as one population. It is the same span as the steady-tick read and
the health bracket. Note that the returned `tail_n` is `steady_filter.py`'s own
`total > 300 ms` count, not a runner's close-tail convention.

**Spec effective rate — `scripts/probe_h20_arm_read.py health`.** Acceptance is a
per-arm *delta* and the tick log carries no accept counters, so the pair of
`/health` reads that bracket the arm is the instrument. The counters are
cumulative and warmup moves them: the two reads must bracket the arm, and summing
across arms double-counts.

```sh
python3 scripts/probe_h20_arm_read.py health <before.json> <after.json>
```

`accept_len` is `Δtokens_generated / Δdecode_forwards` — the engine's own forward
counter. **Never divide drafted tokens by `--depth`**: at depth 3 the drafted
delta is already 3x the forward count, so the depth is read out of the counter,
not supplied. The two rates a report must carry:

- **tick rate** = 1000 / steady-body tick median (per forward);
- **effective tok/s** = `(decode ticks + accepted) / warm decode seconds` — the
  accepted bonus tokens coalesce into the verified token's chunk, so a chunk
  counter sees only the verify forwards and understates the arm.

**Self-consistency, not an instrument.** `Δdecode_forwards` over the arm's health
bracket must equal the number of tick lines parsed from the same span. A
mismatch means the bracket and the log window are not the same arm (a restart
inside the bracket resets the engine counters, which `health` reports as
`straddled_restart`). Check it before any number leaves the session.

**Cold tier: four keys, and report the fill state with the rate.** The occupancy
is the sum over both tiers × both pools:
`kv_cold_private_bytes + kv_cold_shared_bytes + kv_cold_private_ssd_bytes +
kv_cold_shared_ssd_bytes`. A RAM-only gate can never pass on the SSD-heavy H20
shape, and a full tier serves 1.4–2.95x slower than an empty one, so a tok/s
number without its cold fill state is not comparable to anything. All four keys
are required: omitting one passed a fill that looked full because it sat on one
side, which has already happened twice on this matrix.

**Shared spill does not survive a boot.**

- **Two files.** `_shared_ssd_path` derives `.prefix.bin` as a *sibling* of
  `--cold-ssd-path`; the private cold data stays in the `.bin` passed in. They are
  the same spill's two halves, not alternatives.
- **Reopen restores no index.** `ColdSsdFile` sizes a reopened file from its
  on-disk length and maps it, but `_slot_of` starts empty and the live extent list
  is zero, so `HostKvPages._shared_ssd_bytes` is 0. Nothing reads an index back
  from disk (`KvBootStore` is a separate mechanism). The on-disk high-water buys
  only a skipped `ftruncate`.
- **The hit path is memory-only.** `share_take` consults the in-process
  `self._shared` dict first and returns None unless that dict already holds the
  key; its SSD read is reach-through for a page this process spilled. A page left
  by a *previous* process has no in-memory record, so it is not reachable at all:
  the new process overwrites from empty slots.

So each arm fills its cold tier **inside its own serve process**. Never assume the
cold tier is continuous across boots.

**There is no four-key plateau — do not wait for one.** With the spill cap off
(`_prefix_spill_bounded()`'s default) and every request an independent 32k, the
shared spill is append-grown and admission-unbounded: the total rises to disk-full
and never levels. A rule that waits for the total to stop moving waits forever,
and a three-key sample sees the *host* tier pin at 8 GiB and calls it a plateau —
the fourth key (`shared_ssd`) is still climbing underneath (A1 is exactly that
case).

The reproducible boundary is the **host tier filling**, because its cap is fixed
at `SERVE_KV_COLD_BYTES` and identical in every arm. So an arm's fill is measured
as two phases, and both are reported with the numbers:

- **host-unsaturated** — `kv_cold_shared_bytes` still climbing;
- **host-saturated / SSD-only** — `kv_cold_shared_bytes` at its cap
  (`≥ 0.995 × SERVE_KV_COLD_BYTES`, i.e. ≈8 GiB) **and** `ssd_mmap` first appears.
  Both markers together; the counter alone is the trap above.

An arm is ready to measure once the host tier is saturated **plus two more
requests** confirming it is in the SSD-only phase. That is a bound a bounded disk
can actually reach. If an arm finishes its 30 requests without saturating the host
tier, report that arm as unsaturated — do not extrapolate a number from it.

Record which phase each arm's numbers came from. `ssd_mmap`, the close tail and
raw tok/s all move with it, so a comparison across the boundary is not a
comparison.

**Wall time is not a rate.** `probe_h20_train_client.py` reports wall per request
raw and does not fold it into tok/s: a prefix-cache hit removes the prefill
entirely (measured on H20: 335.7 s cold vs 3.3 s on a hit, same prompt), so a wall
median over a batch that mixes hits and cold prefills is a median of two
different things. Group by hit / fresh-prefill before quoting one.

**A tail count belongs to the tool that printed it.** `steady_filter.py`'s tail
is `total > 300 ms`; a runner's own close-tail convention is a different count.
Do not put two tools' tail numbers in one column.

## The graph-off arm: the verdict is the engine's report, not the banner

**This section describes the behavior after #762 lands** (the ops hotfix for the
#758 regression). Written 2026-09-20; if you are reading it much later, check that
#762 is in before assuming the launcher passes an explicit flag either way.

**Fixed rule, do not relax it:** an arm's graph mode is read from the engine's own
report, never from the launcher. The `arm: … decode_graph=` banner is a log cut,
kept for grepping.

The reason is a live bug, not a hypothetical. `--decode-graph` is a `store_const`
flag (`const=True`, `default=None`) and `None` is **AUTO**, which on sm90
resolves to *captured*; the off switch is a separate `--no-decode-graph`. A
launcher that turns the graph off by merely *omitting* `--decode-graph` therefore
boots a graph-**on** engine while its banner reads `decode_graph=off` — the
self-certification line lied, and the arm measured as graph-off was the graph-on
control. Shipped #758 regressed exactly this way; #762 (merged 2026-09-20) fixed
the launcher to pass an explicit flag either way. **Before #762, A2 and A4 must
not be read as graph-off arms.** After it, banner and engine agree — but the
engine-reported rule stays, because a banner that *can* disagree is not evidence.

**Two criteria, both engine-reported:**

1. **`/health`'s `decode_graph`** while the arm runs — `engine._decode_graph_on`,
   the *resolved* value, not the requested one.
2. **The boot line `tilerl serve: N decode graphs in Ns`** (printed by
   `cli.py`'s serve path from `engine.precapture()`) — `0` is off, `8` is on.
   `precapture()` returns 0 immediately when the graph is off, so the count is
   the engine's own, printed before any traffic.

Two candidates were tried and are both wrong:

- `blocks_total` is **invariant** to the graph. `usable_blocks` is
  `num_blocks - (pad_block is not None)`, i.e. capacity is deliberately *net* of
  the row the captured path adds, so the same shape reports the same number
  either way (measured: A2 and A1 both 4425).
- the `4x2k` warmup line is printed **unconditionally** by
  `serve_warmup_hybrid.py` — four concurrent 2k requests, graph or no graph. Two
  boots that *both* ran the graph measured 311.9 s and 67.4 s on it: TileLang JIT
  warm/cold, not graph-vs-eager.

The banner is a log cut, a warmup timing is not a discriminator, and neither is
a pool size. Read `/health`.

## Arms

One serve env delta per arm against `scripts/serve_h20.sh`'s defaults. All arms
also carry `LIVENESS_POLL_S=999999` and the default cold tier
(`SERVE_KV_COLD_BYTES` / `SERVE_COLD_SSD_BYTES` = 8 GiB each) except A5. Those are
the *budgets*; what an arm actually holds is read per arm at run time, and it is
not continuous across boots — see the cold-tier rule above.

| arm | serve env delta | question |
|---|---|---|
| A1 | — | the reference: d1 / k128 / graph-on / W=0 |
| A2 | `SERVE_DECODE_GRAPH=0` | the graph's contribution at d1 (certify from `/health` + the boot graph count) |
| A3 | `SERVE_DEPTH=3` | depth alone, graph-on |
| A4 | `SERVE_DEPTH=3 SERVE_DECODE_GRAPH=0` | depth × graph interaction |
| A5 | `SERVE_SPARSE_K=0 SERVE_COLD_SSD=""` | the dense engine, no sparse, no cold tier (#759 merged) |
| A6 | A1 + `SERVE_DRAFT_WINDOW=2048` | the V100 W=2048 arm, cold-filled, V100 wall protocol |

### Every arm fills the cold tier before it is measured

Because the spill does not survive a boot (above), each arm's order is fixed:
**boot → fill inside that process until the host tier saturates + two confirming
requests → then run the n=30 warm**. Fill-phase ticks are discarded, exactly as in
the V100 `fill2/warm2` protocol — this is that protocol's shape with a reachable
target, not a new one. The fill count is not a constant to copy: it is whatever
crosses the boundary on that arm, read from the four keys while filling.
`probe_headroom_coldtail.py arm` performs this shape for A6; for A1–A5 the fill is
the arm runner's own step before the corpus pass.

An arm that never saturates says so rather than reporting a rate — a
cold-sensitive number taken mid-fill is a number from an unknown condition.

### A5's cold tier is off for a different reason than k=0

`SERVE_SPARSE_K=0` alone does **not** disable the cold tier. The engine attaches
it whenever `kv_cold_bytes` is nonzero, independent of `sparse_k`, and
`serve_h20.sh` defaults `SERVE_KV_COLD_BYTES` to 8 GiB. What actually keeps it
off is `SERVE_COLD_SSD=""`: the launcher then never expands `COLD_ARGS`, so
`--kv-cold-bytes` is absent and falls back to the CLI default 0, and no cold tier
is attached. The arm therefore pins **both** — setting `SERVE_SPARSE_K=0` while
leaving the spill default on would boot the dense engine *with* an 8 GiB host + 8
GiB SSD tier it never fills. `--dry-run`'s `cold_ssd=<disabled>` is the self-cert.

### A6 has a hard prerequisite

A6 drives `SERVE_DRAFT_WINDOW`, which only exists on the launcher once the
`SERVE_DRAFT_WINDOW` pass-through lands. Before that, `serve_h20.sh` passes no
`--draft-attn-window-tokens`, so an "A6" boot is silently the full-prefix arm.
Confirm the flag is in `/health`'s resolved config, or in `--dry-run`'s argv,
before trusting the arm.

## Client protocol

**A1–A5** use `scripts/probe_h20_train_client.py` — the V100 corpus convention on
the wire (`corpus.py` wikitext-103 **train** stream, disjoint spans via
`tiled_spans(skip=512)`), one `/v1/chat/completions` per span, bracketed by two
`/health` reads that `probe_h20_arm_read.py health` consumes:

```sh
python3 scripts/probe_h20_train_client.py --url http://127.0.0.1:8000 \
    --ctx 32768 --n 30 --gen 64 --split train --out <tree>/runs/<arm>_32k_n30.json
```

**Merge order:** this client is #761, which is approved but not on `main` at the
time of writing. An A1–A5 arm cannot run until it lands, because the headroom
probe cannot stand in (see below). `probe_h20_arm_read.py` is #759, which **is**
on `main`.

It is not `probe_headroom_coldtail.py`: that probe talks to a live serve but
synthesises its 32k prompt (a uuid lead plus a word stream), so it cannot answer a
question whose comparison partner was measured on wikitext. It is not
`probe_draft_window_sweep.py` either: that builds its own engine, which would
collide with the serve under measurement.

**A6 is the exception** — it mirrors the V100 wall protocol (fill the cold tier
full, then measure warm decode), not the n=30 corpus pass, so it uses
`probe_headroom_coldtail.py`:

```sh
python3 scripts/probe_headroom_coldtail.py arm \
    --url http://127.0.0.1:8000 --headroom 0 --log /work/serve_h20.log \
    --out <arm>/arm.json --prompt-tokens 32000 --fill-n 2 --warm-reps 2 \
    --expect-window 2048
```

`--warm-reps 2` refills and re-warms per rep, so the rate is a median with a
spread. Its `p50` is the **wide** set (`dec > 0`, including captured-graph and
close-tail ticks); re-run `scripts/steady_filter.py` on the same log to place it
on the standard set before it meets an A1–A5 number in a table.

## A1 as the worked example

The reference arm, d1 / k128 / graph-on / W=0, 32k, **window: the ticks at and
after `n >= 461`** (the arm opens with a 2-sample probe whose ticks are a
different cold state; the whole file counts 65 extra ticks and must not be used).

Tick level, `steady_filter.py` on that window
(`{"steady_n": 992, "steady_p50_ms": 85.0, "steady_p90_ms": 104, …,
"tail_n": 28, "tail_max_ms": null, …}`):

| field | value |
|---|---|
| steady ticks (n) | 992 |
| tick median | 85.0 ms → **11.8 tok/s** per forward |
| p90 | 104 ms |
| model-segment median | 77.0 ms |
| close tail | 28 ticks, max 208 ms (the runner's count; `steady_filter`'s own `total > 300 ms` tail is empty) |

Spec effective rate, `probe_h20_arm_read.py health` over the n=30 bracket
(`{"d_accepted": 912, "d_drafted": 992, "d_generated": 1920,
"d_decode_forwards": 992, "accept_rate": 0.9194, "accept_len": 1.9355,
"straddled_restart": false}`):

| field | value |
|---|---|
| accept rate | 0.9194 — **superseded by A1r** |
| accept_len | 1.9355 generated per forward — **superseded by A1r** |
| Δ decode forwards | 992 (= the steady tick count — the self-consistency check passes) |
| effective tok/s | **22.8** — **superseded by A1r** |
| cold phase | **host-saturated, SSD-only** — host 8 GiB full, `shared_ssd` 59.5 GiB and still climbing, cap off |

**A1 was not taken at a plateau** — there is no plateau in this configuration
(above). Its tick reading is taken from the host-saturated / SSD-only phase, and
the phase is part of the number:

- **The median does not track cold growth.** The 30 per-request p50s sit in
  80–92 ms with no trend against request index (r ≈ −0.2), so `11.8 tok/s` stands
  as a reading of that phase.
- **The tail is set by the cold tier.** The per-request maxima hold near
  1200–1346 ms from the second request on — the phase where `ssd_mmap` first
  appears (52 of the 992 steady ticks carry a nonzero `ssd_mmap`, the first at
  tick 935). This is the same effect as the 2026-09-17 cold-full entries
  (finalize relocation at 32k), not a contradiction of it.

So a tick **median** is quoted with its phase; a **tail or p90** is quoted with
the cold tier's state. A1r reports its own phase.

**Why those three are superseded:** the client did not pin `enable_thinking:
false`, so this run measured a think-**on** token distribution and its acceptance
and effective rates carry that. The tick rate is unaffected — the 30 per-request
p50s land in 80–92 ms, a 12 ms spread, so generated content does not move tick
time — and the tick row above therefore stands. Re-run the client with thinking
off (A1r) before quoting acceptance or effective tok/s.

Raw artifacts: `serve_h20_full.log`, `a1_32k_n30.json`, `cold_trace.txt` (the A1
runner's `/tmp/p2rev/a1_artifacts/`; vendor into the tree when the matrix lands).
Do not quote A1 as the n=2 probe's 80 / 71 ms — that window is a different cold
state, and the difference *is* the cold-tier fill. Model segment 77 ms against
the V100's 168 ms is 2.18x, but the two ran different W, so the comparison is
labeled, not bare.

## Rule

An arm matrix is only a matrix if every arm's env is pinned the same way, every
number names the instrument that produced it, and the arm's own boot certifies
which one ran. A banner is a log cut; the resolved value is the evidence. A rate
without its cold fill state is not a rate. And the delta a report quotes is
`/health`'s, read over a span whose tick count agrees — if the two disagree, the
span bracketed a restart and the number is not an arm.
