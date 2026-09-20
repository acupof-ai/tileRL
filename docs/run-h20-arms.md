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
   the engine's own, printed before any traffic. At `--depth 3` the count is
   **16**, not 8: the bucket set scales with depth.

A third reading is available, and it is the **allocation result** rather than the
engine's own boolean — so the two can falsify each other:

3. **`/health`'s `memory` rows: the `kv_pool` note's block count.** The pool is
   built `num_blocks + pad`, and the note reports that **gross** figure, so it
   carries the graph's padding row: **`4426 blocks` is graph-on, `4425` is
   graph-off** (measured at **d1** on two independent graph-on boots and one
   graph-off). The paired `state_slots` `derived` bytes separate too — the padding
   row holds a slot as well as a block, **476 MiB apart at d1**, and at this
   configuration that state-slot is the bulk of the cost against a KV block's
   ~1 MiB.

   The count is a **d1 observation**: at `--depth 3` the draft layer count differs,
   so the absolute figures are to be read per arm rather than assumed. What must
   hold at any depth is the **relation** — graph-on's gross KV-pool blocks are
   graph-off's **plus exactly one** (the padding row). Check it before taking an
   arm's numbers: if the two arms' gross counts differ by anything else, the
   configuration is wrong and the arm is not readable.

Note this is the *gross* figure, and the reason the caution below is about
`blocks_total` specifically: `blocks_total` is the **net** `usable_blocks`, which
subtracts the same pad, so it is invariant. Two views of one pool, one
discriminating and one not — read the `memory` note, not `blocks_total`.

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

### A6 fills the cold tier first; A1–A5 start cold

The spill does not survive a boot (above), so an arm's cold state is built inside
its own process. **A6 gets there before it measures**, because it mirrors the V100
wall protocol: fill, then measure warm, `probe_headroom_coldtail.py arm` performs
that shape and reports a plateau rate.

**A1–A5 do not fill.** They run the n=30 corpus pass from a cold start, and the
against-A2 comparison is taken **per phase**, not over the whole arm. The reason is
that filling first would delete the phase the comparison needs.

The cold tier's four-key total has **no plateau** on these arms: with the SSD
budget effectively unbounded and one independent 32k request at a time, it grows
until the disk fills. What *is* reproducible across boots is the moment the **host
budget saturates** — `kv_cold_shared_bytes` reaches its 8 GiB cap, after which cold
grows only on the SSD side. That is a fixed configuration point, so every arm
crosses it at the same occupancy, and it splits each arm into two phases:

| phase | condition | meaning |
|---|---|---|
| **A host-filling** | `shared < cap` | cold is still being absorbed in RAM |
| **B host-full** | `shared >= cap` | RAM full; only the SSD tier grows |

**The split is read from the log, not from a sampling threshold.** The host cap is
reached between two 10 s samples of the cold trace, so a `shared >= 0.995*cap` test
on that trace lands a whole request late (measured on A2: the sample before the
crossing reads `shared=6.750, ssd=0`, the one on it reads `7.935 / 1.065`, and the
threshold fires one request after that). The **action count is the boundary**: the
first steady tick whose `ssd_mmap > 0` — the SSD tier engaging is the same event,
counted where it happens. On A2 and A1r that boundary falls inside arm request 3
(A2 tick 846, A1 tick 935); on A1 it is tick 935.

A phase boundary in the *cold trace* maps onto a tick index by `decfwd`, the one
field the trace shares with the serve log's tick counter — and it is an identity,
not an estimate: on A2 every one of the 30 request boundaries satisfies
`cumulative steady ticks == Δdecode_forwards`, diff 0, total 1028 = 1028. Nothing
else in the trace is in tick coordinates; a slice taken on sample index cuts
inside the wrong request.

Compare arms **phase B against phase B** (the sample is the larger one: A2 n=884 of
996) and report phase A beside it. A whole-arm p50 mixes the two cold states and
must not meet another arm's phase number in a table.

An arm whose host tier **never** saturates is the anomaly and says so: an 8 GiB cap
under a single 32k stream is reached every time, so failing to reach it means the
arm did not run the configuration it claims.

### The tick p50 band is ±3 ms, and bootstrap alone understates it

Deciding whether two arms differ on tick p50 needs a band that was measured, not
assumed. Three layers, only the middle one of which resampling can correct:

| layer | estimator | measured halfwidth |
|---|---|---|
| sampling error | iid tick bootstrap (n≈1000) | **0.00 ms** |
| within-request correlation | **bootstrap whole requests** (30 blocks) | **1.00 ms** |
| state drift | same-boot, different-phase control | **5 ms** |

The iid bootstrap returns **0**, which is false: ticks inside one request share a
page-residency state (32–37 ticks per request here), so resampling them as
independent units collapses the variance to nothing. Take the request-block figure.
Cross-check by splitting odd/even requests: A1 differs 0.0 ms, A2 2.0 ms.

Resampling cannot see the third layer at all. On A1 the same boot's warmup probe
(2 requests, cold tier still near empty) measured p50 = 80 ms against the arm's own
85 ms — a 5 ms drift with no configuration change. The band is therefore **±3 ms**,
not the bootstrap's 1.

**The decision is pre-registered, so the result cannot pick the rule afterwards:**

- A1r in **82–88** → think has no effect on tick; the A1-vs-A2 11 ms gap is
  attributed to the capture pool's memory layout, and the fourth cell is not run.
  88 is the boundary and falls on the no-effect side: the mechanism prior is that a
  decode forward's per-token work does not depend on content, and evidence against
  a prior has to be stronger than the boundary.
- A1r in **93–99** → content-driven; run the fourth cell (think-ON, graph-OFF) to
  fix the interaction.
- A1r in **89–92 or outside both** → the two effects are not separable; run the
  fourth cell.

Acceptance rate needs no fourth cell: the graph does not execute at 32k sparse on
either arm (A1's `path=graph` ticks are 20, all before the steady span and all
non-sparse), so the A1 `.9194` (think-ON) against A2 `.8551` (think-off) gap can
only be the thinking knob.

### The graph's cost here is memory, not a tick path

A1 (graph-on) and A2 (graph-off) both run 32k sparse decode **eagerly**, yet
graph-on holds consistently more memory:

| quantity | A1 graph-on | A2 graph-off | delta |
|---|---|---|---|
| driver `device_free` | 52.886 GiB | 54.560 GiB | **+1.67** |
| in-tick allocator `free` | 54149 MiB | 55593 MiB | +1.44 |
| in-tick `reserved` | 42746 MiB | 41506 MiB | −1.24 |

The cause is `decode_graph.py`'s `ensure_pad`, which reserves the padding row when
capture is set up. **That is a capture-time cost, independent of whether the graph
is ever replayed** — which is why it shows up on an arm whose sparse decode never
enters the captured path. Report it as its own finding; it does not need the tick
attribution to stand.

**`--decode-graph` also changes the eager sparse tick's code path, and that is a
separate fact from the memory above.** `build.py` resolves
`sparse_device_select = _graph_on(backend, decode_graph)` whenever the caller did
not set it, and the serve sets neither this nor the CLI's own switch; on sm90
`_graph_on` is True. So **turning the graph on silently turns sparse device
selection on**, and a purely-decode tick then runs the device-resident-table path
(`SparseRuntime` → `_init_device_tables`, fixed-width capture-ready buffers, a
re-score every 8th tick) instead of rebuilding the packed `[selected; own]` table
and re-resolving logical→physical per row, per group, per plane on the host.

**Read the two together.** The graph flag carries a path change *and* a memory
change; neither is evidence for the other, and the memory cost does **not** mean
the path is slower — on the A1r/A2 single-variable contrast (same tree, same
thinking setting, `decode_graph` the only difference) the **graph-on** arm's clean
phase-B median is **14 ms lower**. Whether that 14 ms is the whole of the path
change's contribution is what the `--depth 3` pair is there to replicate: if d3
reproduces the same sign, the magnitude is stable; if it reverses or vanishes, the
branch is still there but the d1 magnitude is a single boot's observation.

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

**Both probes are on `main`** (`probe_h20_train_client.py` = #761, merged; the
arm reader = #759). An A1–A5 arm needs the client, not the headroom probe, and the
headroom probe cannot stand in (see below).

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
| tick median, phase B (host-full) | **85.0 ms** (n=836 body) |
| tick median, phase A (host-filling) | **86.0 ms** (n=128 body) |
| p90 | 104 ms |
| model-segment median | 77.0 ms |
| close tail | 28 ticks, max 208 ms (the runner's count; `steady_filter`'s own `total > 300 ms` tail is empty) |

The two phase medians are 1 ms apart on this arm, which is why A1's whole-arm 85.0
is quotable at all — but A2's two phases are **92.0 against 96.0**, 4 ms apart, and
phase A there is only 112 body ticks. Compare phase B against phase B. `n >= 461`
and the 992 above are the same window: the two 2-sample-probe request groups are 65
ticks, and 1057 − 65 = 992 = Δdecode_forwards.

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
  as a reading of that phase. Phases A and B differ by 1 ms (86.0 vs 85.0), so on
  this arm the phase label does not move the median — **it does on A2** (92.0 vs
  96.0), which is why the label is mandatory rather than decorative.
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

## The graph flag as a result: five arms, and what replicated

A1–A4 ran 2026-09-20, n=30, 0-start, one serve per boot. `clean` below is the
steady-B median over the `ssd_mmap == 0` subset — the verdict column, since
cold-relocation ticks sit inside the same steady set and their count differs per
arm (A2 carries 102, A1r 56). Every arm's `Δdecode_forwards` equals its window's
steady tick count, diff 0.

| arm | depth | graph | steady | boundary | **clean B p50 (n)** | B p90 | model | accept | tok/fwd | tick tok/s | eff tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 1 | on | 992 | 935 | **85 (811)** | 103 | 77 | — | 1.9355 | 11.76 | 22.77 |
| **A1r** | 1 | **on** | 1019 | 866 | **82 (826)** | 95 | 74 | 0.8695 | 1.8842 | 12.20 | 22.98 |
| **A2** | 1 | **off** | 1028 | 846 | **96 (809)** | 106 | 81 | 0.8551 | 1.8677 | 10.42 | 19.46 |
| **A3** | 3 | **on** | 639 | 797 | **100 (514)** | 119 | 85 | 0.6641 | 3.0047 | 10.00 | 30.05 |
| **A4** | 3 | **off** | 642 | 795 | **119 (497)** | — | 97 | 0.6651 | 2.9907 | 8.40 | 25.13 |

**The single-variable contrasts.** A1r/A2 differ only in `decode_graph` (same
tree `96040093`, same thinking setting, same 0-start). A3/A4 differ the same way
at depth 3:

| pair | contrast | ms | % |
|---|---|---|---|
| d1 | A1r(on) 82 vs A2(off) 96 | **−14** | **−14.6%** |
| d3 | A3(on) 100 vs A4(off) 119 | **−19** | **−16.0%** |

**Both signs and both magnitudes replicate.** Graph-on is *faster* on the 32k
sparse decode tick, by about 15% at either depth. Quote the **percentage**: the
absolute gaps differ (14 vs 19 ms) only because the d3 baseline is higher, and
that is the reason both are reported.

The band was fixed before A4 ran and was not moved after: request-block
bootstrap halfwidth ≤1.5 ms and split-half drift ≤3.0 ms on every arm, all under
the pre-registered ±4 ms. A4 landed 15 ms clear of the nearer threshold.

**The mechanism is a code path, not the graph replaying.** `build.py` resolves
`sparse_device_select = _graph_on(backend, decode_graph)` when the caller leaves
it unset, and the serve sets neither that nor the CLI's switch — so on sm90
**turning the graph on also turns sparse device selection on**, and a
pure-decode tick runs the device-resident-table path instead of rebuilding its
packed table and re-resolving logical→physical on the host, per row, per group,
per plane. Neither arm replays a graph at 32k sparse: `sparse_min_tokens` is set,
so `sparse_graph_on` is False and both decode eagerly. What the flag buys here is
the device-select path, which is why the effect is a tick *rate* and not a
latency spike.

**Two costs the same flag carries, which are not evidence about the tick.** The
capture-time `ensure_pad` reservation is resident on graph-on arms: driver
`device_free` 52.886 vs 54.560 GiB, allocator `reserved` 42746 vs 41506 MiB. And
the padding row holds a **state slot** as well as a block — at d1, 476 MiB of
state against a KV block's ~1 MiB. **The d1 state figure does not extrapolate:**
d3 measured **754 MiB**, so read the pad's state cost per arm rather than
scaling it. (A prediction scaled from d1's 454 MiB to d3 was wrong, which is why
it was demoted from a gate to an observation.)

**A4 ran without a cold trace.** The previous sampler had stopped (13:19, ~40 min
earlier than its own `sleep`-based estimate — its real period exceeds its sleep
interval) and none was started for A4, by ruling: attaching a sampler mid-window
would be a second lifecycle misalignment, and it would only catch the tail of the
fill. So A4's phase boundary comes from the tick lines' own `ssd_mmap`, and its
cold state from the closing `/health`. **Its fill curve is missing and the A/B
split has no trace corroboration** — the tick measurement does not depend on it.

## Rule

An arm matrix is only a matrix if every arm's env is pinned the same way, every
number names the instrument that produced it, and the arm's own boot certifies
which one ran. A banner is a log cut; the resolved value is the evidence. A rate
without its cold fill state is not a rate. And the delta a report quotes is
`/health`'s, read over a span whose tick count agrees — if the two disagree, the
span bracketed a restart and the number is not an arm.

A contrast needs its variable to be the *only* one: two arms that differ in depth
and graph at once are not a contrast, and two that ran different trees are not
one either — A1 sits on an earlier tree than A1r/A2, so its `model` column is
labelled, not compared. And a checklist's items can all be true while the list
answers the neighbouring question: verify the *resolved* flag, not the requested
one; verify the flag's *identity*, not that its substring appears; and before
sending traffic, count the client processes, because "this serve is configured
correctly" and "nothing is already running on it" are different questions.

