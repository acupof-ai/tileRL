# A full cold tier makes sm70 sparse 32k decode long-tail — 2026-09-17

> Status: **open, measured, one candidate fix rejected on device.** The
> `--device-headroom-mib` shrink path was run to verdict on V100 on 2026-09-18
> (four arms 0/384/768/0) and is a **no-op for this tail** — see Device verdict.
> The host cold tier's own O(n) LRU/spill cost was fixed in #735 (pending device
> confirmation); the dominant remaining suspect is the per-request close-time
> lock-held page transfer, awaiting a #732 five-segment device attribution
> before a fix is chosen. Surfaced during the draft read-window W=2048 sign-off
> (#698) but independent of it. Related: #654/#656, and the #665 sm70 sparse
> tick timing split.

## Context

V100 sm70 hybrid serve, main `07eca439`, draft read window W=2048, flags
`--kv-cold-bytes 8589934592 --cold-ssd-bytes 8589934592`, sparse k=128,
d1, sparse-min-tokens 8192, `TILERL_STEP_TIMING=1`. The 32 GB card holds
~31.3–31.6 GB live; idle `device_free` is only ~200–480 MB.

A primed (warm) 32k sparse request decodes far below the ~9 tok/s the steady
tick predicts: 1.4–2.95 tok/s with the cold tier full, **7.86 tok/s** on a
fresh boot with an EMPTY cold tier. The median decode tick is healthy in both
states (~175–190 ms); the entire loss is a heavy long tail whose frequency
tracks cold-tier fullness.

Within one warm 32k run, consecutive ticks at the same `cmax=1916 own_w=8`
split bimodally: `model=157–168 ms` on most ticks, `model=1169–1259 ms` on
others. The same cmax bucket at two speeds rules out attention growing with
cmax, cmax-bucket recompile, sparse_select (1–10 ms), draft_step (7–45 ms
under W=2048), and offers publishing for the worst ticks.

## Root cause — two distinct long-tick mechanisms (2026-09-17 observation)

**1. `sparse_finalize` batch page moves (dominant, scales with fullness).**
A tick that moves/drops pages pays `sparse_finalize = 629–667 ms` for
`offers_pages=129–144` (one smaller event 186 ms / 72 pages; empty-tier run
81 ms / 112 pages). The same ~120–144-page batch costs **7–8x more
(81 → 629 ms)** once the cold tier is full and device_free is ~200 MB.

**2. "Hollow" forward ticks (unattributed).** `total=1.1–1.4 s` with the
forward envelope ~1.1 s but its measured inner sum only ~0.3–0.5 s
(`model` ~330–520 ms), `sparse_finalize` 1–3 ms, `offers_pages=0`. Time is
inside the model forward but uncovered by any segment — an implicit GPU sync
or allocator stall.

Two-phase isolation (same boot, same W=2048; only cold-tier fill changed):

| phase | kv_cold_shared | device_free | warm 32k tok/s | tick p50 | p90 | max | >300 ms |
|---|---|---|---|---|---|---|---|
| 1 empty | 0 → 2.3 GB | 485 MB | **7.86** | 176 | 286 | 909 | 1/10 |
| 2 full | 8.07 GB | **206 MB** | **4.25** | 175 | 790 | 958 | 2/5 |

Two further full-tier samples: C0 6.56 tok/s (p50 180, 2/10 >300), C1 2.69
(p50 393, p90 1337, 7/12 >300). Across ~27 full-tier decode ticks the median
stays ~175–190 ms while the >300 ms fraction is 40–58 %.

## Device verdict — `--device-headroom-mib` is a no-op here (2026-09-18)

The shrink-pool candidate (#702; build the sparse main pool smaller so target
MiB is free after both pools attach) was run on tree `acbf09fa` with
`scripts/probe_headroom_coldtail.py`, W=2048 fixed on every arm, a full ~8 GiB
cold tier, one fresh boot per arm, four arms `0 → 384 → 768 → 0` (the trailing
0 is a clean cold-start bookend). raw = 17-token first-token→last-token; steady
p50 counts only `dec=1 & sparse=1` warm ticks (idle `path=graph/sparse=0`
~29 ms ticks excluded).

| arm | device pool | free @ full (MiB) | raw tok/s | steady p50 ms (n) | close stall inside 17-tok span | fill finalize p50/max ms | fill offers p50/max pages | fill wall |
|---|---|---|---|---|---|---|---|---|
| H0 (pre) | 2213 / -0 | 228 | 4.948 | 184 (17) | 6.35 s outside | 190 / 931 | 106 / 175 | ~29.3 min |
| H384 | 2213 / -0 | 28 | 3.233 | 186 (17) | 4.99 s inside | 197 / 729 | 109 / 175 | ~29.8 min |
| H768 | 2213 / -0 | 142 | 3.787 | 186 (17) | 7.05 s inside | 205 / 1138 | 109 / 175 | ~31.1 min |
| H0 (tail, cold start) | 2213 / -0 | 692 | 3.755 | 185 (17) | 5.43 s inside | 222 / 979 | 109 / 175 | ~30.9 min |

**The knob does not engage at this configuration.** `sparse_pool_fit_headroom`
measures `free_bytes` after weights + recurrent state are resident but before
any KV pool (`memory.py:121`, `build.py`); that margin is ~4487 MiB. The 8 GiB
cold tier is host pinned RAM (`HostKvPages`) filled at runtime, never a
subtrahend, and device finalize only allocs/frees within the fixed
`num_blocks` free-list pre-reserved at build — it does not ask the cuda
allocator for new VRAM. Since 4487 ≥ 768 the solver returns the ceiling
unchanged: every boot prints `main pool 2213 blocks (-0) predicted free 4487
MiB`, and the three arms' pools are byte-identical.

- Steady decode p50 is 184–186 ms on all four arms (≈5.0 tok/s, sm70 per-tick
  compute; the pre/post 0 bookends agree). Headroom changes nothing.
- Fill `sparse_finalize` is the same distribution on all four arms (p50
  190–222 ms, max 729–1138 ms; offers p50 106–109 / max 175), with no
  monotone trend in the knob. Today's build is heavier than the 2026-09-17
  note (max 931 > 667, batch max 175 > 144).
- `sparse_headroom_dropped_blocks` = 0 on every arm.
- raw tok/s spans 3.23–4.95 for one reason: whether the deterministic
  per-request close-time stall lands inside the fixed 17-token first→last
  span. It is unrelated to headroom and to outside traffic — the clean tail-0
  arm (no other client) reproduces it. Report the **tick median**, not raw, as
  the steady rate.

**Verdict: headroom is rejected for this tail.** Do not enable it to chase the
full-tier finalize long tail; default stays 0. A build-time VRAM knob cannot
relieve a cost that runs in host RAM and inside a pre-reserved device
free-list. (If "trade concurrency for headroom" is ever wanted on its own
merits, the solver would have to subtract the device peak transient working
set of one maximal finalize batch — not the 8 GiB host tier — bounded by a
worst-case constant; the single-slot floor 554/slot may make that infeasible
on V100. Not pursued: it does not fix this defect.)

## The close-time lock stall — now separated (the former "hollow" / raw jitter)

The ≥1 s stalls are deterministic and per-request, not hollow GPU forwards.
Each request end logs exactly one `dec=1 model≈165 ms sample=<1.8–8.9>s` tick
at the decode→next-prefill boundary (6 per boot: 5 fill + 1 warm; observed
1934/1802/3007/4651/5440/4988 ms on H384, max 8.9 s on H768). The `sample`
bucket in `step()` wraps `_sample_batch → _verify → _commit → _finish →
_release` under the engine `_lock`; the stop token finishes the request and
`_release` transfers the still-resident pages to the shared prefix store one
page at a time (per-page D2H syncs, an unpinned-churned blob copy, and
per-page SSD read+write for already-spilled pages). The next prefill's admit
and the client poll contend for the same `RLock`, so the user waits this 1.8–
8.9 s even though token emission is finished.

Byte volume does **not** explain 5.4 s (trunk f16 2.0 MiB/page ≈ 2.0 GiB for a
32k request ≈ 0.2 s at 12 GB/s; 5.4 s would need ~60 GiB, ~30x). The leading
hypothesis is the accumulated latency of ~10k independent per-page syncs plus
pinned-allocation churn and the O(tier) LRU scan — not transfer bandwidth.
#732 splits the release path into five timed sub-segments
(`pub_bounds_d2h / pub_draft_clone / pub_frame_d2h / pub_share_hold /
pub_ssd_transfer`, env-gated, zero-cost off); the next V100 window attributes
the milliseconds per segment and the fix is chosen from that measurement, not
from a byte estimate. Do not pre-write the batch-D2H change on the bandwidth
story.

## Fixes — status

- **Host cold tier O(1) LRU + zero-copy/extent spill — landed (#735), device
  delta pending-remote.** The fill finalize path also paid pure-host cost the
  headroom knob structurally cannot touch: `_enforce_budget` rebuilt
  `list(_ram_order.items())` and scanned it on every hold once the host budget
  bound (O(resident pages) per hold; stale records re-scanned forever), and
  `ColdSsdFile` copied every page through `numpy().tobytes()` and rebuilt the
  whole mmap per high-water slot. #735 makes eviction an OrderedDict-front
  O(1) pop, writes through an `np.frombuffer(mmap)` view, and grows the spill
  file one 64-slot extent per ftruncate+mmap. CPU: drop-scan 747.5→0.9 µs/hold
  at 16384 resident pages (flat), SSD path 3.90→0.068 ms/hold at R=4096. The
  real ~2 MiB/page device D2H + SSD I/O is not modeled; the V100 fill-finalize
  split is re-measured next window. Wins entry:
  [wins/2026-09-19-cold-tier-olru-zero-copy-spill.md](../wins/2026-09-19-cold-tier-olru-zero-copy-spill.md).
- **Release/close lock stall — open, instrument first (#732).** Fix options
  (bounds row batching, resident-blob single-sync + a pinned staging pool,
  skip force-close when no follower waits, move close out of the step lock,
  lazy SSD lift) are ranked only after the #732 five-segment device read lands.
- **Draft read window — decided separately, no flip.** The W=2048 default
  question ran its own n=30 device sign-off and did not clear the +20 % bar
  end-to-end (32k +15.6 %, 16k +10.4 % with an unclosed bracket): default
  stays 0, 2048 stays opt-in. See
  [errors/2026-09-19-w2048-window-end-to-end-not-significant.md](2026-09-19-w2048-window-end-to-end-not-significant.md).
  The separate draft **decode bucket-width** lever (workflow top 1) is still
  open: a draft decode-only forward can run on the real 1–2 decode query rows
  instead of the S=64 prefill bucket (`spec.py` `DraftHead.step`, predicate
  shared with `_windowed_read_kv`; host-only shape decision, CPU allclose
  gate), cutting accepted-tick draft attention/projection launch width ~30x.
  After the close/finalize fixes land, re-measure W (and bucket width) with
  per-prompt tok/s instrumentation — the close stall currently masks the
  per-forward draft gain.

**Next-step order:** (1) host cold-tier O(1) — **landed (#735)**, confirm on
device next window; (2) run the #732 five-segment device attribution and only
then pick the release-lock fix; (3) take the draft decode bucket-width lever
(top-1 workflow candidate); (4) re-run the W sign-off with per-prompt tok/s.
Headroom stays off (it is a measured no-op); W default stays 0 / 2048
opt-in until (2)–(4) land.
- **Allocator fragmentation** is orthogonal; `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` is already load-bearing on this V100
  ([errors/2026-09-03-expandable-segments-is-load-bearing.md](2026-09-03-expandable-segments-is-load-bearing.md)).

## Measurement discipline (from the 2026-09-18 window)

- **Silence the liveness chat probe for the window.** Boot env adds
  `LIVENESS_POLL_S=999999`. `serve_liveness.py` sends a real chat completion
  every 60 s when a slot is idle; it interleaves into the decode window and
  measured a false 3.23 vs a true 5.31 tok/s steady. Restore 60 s after the
  window — it is the idle self-heal probe, never leave it disabled.
- **Tick selection.** Count only `dec=1 & sparse=1 & model>0 & sample>0` warm
  ticks; drop `path=graph / sparse=0` ~29 ms idle ticks. Also separate the
  one per-request close tick (`dec=1 model≈165 ms` normal, `sample` owns
  multi-seconds) from the steady set — dropping only idle ticks is not enough.
- **Report the tick median as the steady rate.** raw first→last tok/s jitters
  with whether the close stall lands in the 17-token span; keep raw only as a
  whole-turn feel number and label whether the span contains a close stall.
  Always state the cold-tier fill state (empty 7.86 / full ~5).
- **Cold fullness gate is private + shared host bytes**, not the private key
  alone: live private resets to ~0 when a fill request ends while the shared
  pool holds the accumulated ~8 GiB. The probe gates on `kv_cold_bytes +
  kv_cold_shared_bytes` and excludes the separate SSD tier (#731).
- **Client / tree forensics.** A non-probe concurrent client contaminated two
  warm windows. Count unique `ss -tnp` peers (each ESTAB connection prints two
  lines); require zero non-probe clients before the warm POST. Headroom/W ride
  env (never argv), so the hard reads are
  `tr '\0' '\n' < /proc/<pid>/environ | grep TILERL_` for the armed settings and
  `readlink /proc/<pid>/cwd` to confirm the serve is actually running from the
  synced tree (`/home/chenkailun.c/tilerl-v100-sse`), not a stale checkout.
- **Probe dispatch/fields are hermetic even though they only fully run on
  device**: subcommand arity (#730), the private+shared cold gate (#731), and
  the whole-log window evidence (#728) all carry self-check asserts.

## How the run was driven (reproducible)

Probe: `scripts/probe_headroom_coldtail.py` (on main at 2dc88a25: #717 load,
#730 arm/compare dispatch, #731 private+shared cold gate, #701 dec/pre phase
tags, #728 warm reps + window cross-check). One fresh boot per arm, W=2048 on
all, probe runs the arms (it never restarts serve):

```sh
python3 scripts/probe_headroom_coldtail.py arm \
  --url http://127.0.0.1:8000 --headroom <0|384|768|0> \
  --expect-window 2048 --warm-reps 3 \
  --log /home/chenkailun.c/servehybridsse.log \
  --out /home/chenkailun.c/headroom_<arm>.json
# then, after every ARM_DONE:
python3 scripts/probe_headroom_coldtail.py compare \
  --arms pre=…/0r.json half=…/384.json target=…/768.json post=…/0tail.json
```

Per rep the probe refills the cold tier with five independent non-prefix-
sharing 32k prompts, refuses below `--min-cold-gb 7`, streams one warm 32k and
times first→last token, parses decode-only ticks from a byte offset taken at
the warm POST, and checks the engaged window from the **whole log (offset
0)** — a build-time constant the engine prints once per (batch, first-page)
shape, so fill's same-shape decode already logged it and a post-warm search
would false-fail (#728). Fail-closed rc3 (cold under gate) / rc13
(NO-DECODE, TOO-FEW-DECODE-TICKS, WINDOW-MISMATCH, NO-GOOD-REPS); success
prints `ARM_DONE <headroom> <median_tok_s> reps=<good>/<N> W=[2048]`.
Capacity read: one slot hot ceiling 553 blocks (`4·128+8+33`, 2.0 MiB),
4-slot pool `slots·553+1 = 2213`, residency `floor(blocks_total/553)`;
`slots_total` is a separate state-pool count.

### V100 device-window operator runbook (ops-owned: deploy / serve / health / rollback)

Scope split: ops brings the tree up, proves the serve healthy at each headroom
setting, and rolls back. After a serve is healthy ops signals the measurement
owner, who runs the probe and owns the ARM_DONE/tail reading; ops does not run
the measurement or interpret results. `0 / 384 / 768` are scan points.

**Fixed facts (read on the box before the window):**
- Serve tree `/home/chenkailun.c/tilerl-v100-sse` (not `scripts/v100.sh`'s
  default `~/tilerl-v100`). Sync to the target merged main that contains the
  knob (#702), the probe (#717 + #730 + #731 + #728), and the dec/pre tags
  (#701); refuse the window if any is absent from `.synced_commit`.
- Checkpoints read-only: 18/18 shards in `~/models/Qwen3.8-27B-NVFP4` match
  the safetensors index; draft `~/mmlu-assets/model_mtp.safetensors`; cold
  spill `~/sparse_cold_128k.bin` exists. `nvcc` at `/usr/local/cuda-12.4/bin`.
- One-time sync: `git archive` the target sha, scp a tarball, extract in the
  tree (overwrites tracked files only; untracked probes / cold SSD /
  checkpoints are untouched), write `.synced_commit`; verify per-file md5
  remote vs local (0 differing).

**Stop helper (stat, not pgrep):** TERM the supervisor pid (parent of the
`tilerl.cli serve` child), sleep 11, then `ps -o stat= -p <pid>` must report
gone for supervisor/serve/guard (a zombie counts alive); require
`nvidia-smi --query-gpu=memory.used` = 0 MiB and an empty
`--query-compute-apps` before continuing.

**Per-arm serve (identical env, only the number changes):**
```bash
H=<0|384|768>   # control is H=0 explicitly (all arms set the var)
/usr/bin/ssh v100 "rm -f ~/.servehybridsse.fuse; cd ~ && \
  TILERL_DEVICE_HEADROOM_MIB=$H \
  TILERL_DRAFT_ATTN_WINDOW_TOKENS=2048 \
  TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0 \
  LIVENESS_POLL_S=999999 \
  setsid nohup bash tilerl-v100-sse/scripts/serve_hybrid_v100.sh \
    >/dev/null 2>&1 </dev/null &"
```
Headroom/W ride env (the supervisor argv does not pass them; `cli.py` defaults
both flags from their env vars). Per-arm gate before signaling the measurement
owner: `/health` 200, current-boot warmup done, boot first-line sha matches,
a warm idle chat returns 200, `[draft-window] W=2048` present, headroom and
free fields recorded, serve pid recorded, and **zero non-probe clients in
`ss -tnp`**. Do not touch the box during the arm. Between arms: stop helper →
GPU 0 → relaunch with the next `H` (same tree, no re-extract). A new boot
empties the host shared pool, so each arm is a genuine cold refill.

**OOM/de-escalation:** if an arm boot-loops a catchable OOM the supervisor
fuse parks it; TERM the whole group by stat, drop to the next-lower arm (or
0), do not push through. On the 2026-09-18 run 768 built fine (-0) — the
~4487 MiB build margin exceeds all three targets.

**Final rollback + verify:** relaunch with `TILERL_DEVICE_HEADROOM_MIB=0`
(and `LIVENESS_POLL_S` restored to 60 s) on the same synced tree; gate = boot
0 correct sha + warmup done + `/health` 200 with headroom 0 + one real decode
smoke (chat max_tokens=8 → 200) + non-zero d1 acceptance.

**Health patrol:** silence the normal /health watcher for the window downtime
(non-200 is expected while arms restart); re-arm after the final baseline is
verified. All pids via `ps -o stat=`; GPU via `nvidia-smi`, never pgrep alone.

## Rule

On a card run within a few hundred MB of full VRAM, a KV cold tier that fills
to capacity turns a healthy median tick into a long-tail serve: the steady
median can read fine while 40–58 % of ticks stall. Judge capacity by the tail
under a **full** tier, not a freshly-booted empty one, and separate the three
costs the tail mixes — fill-time host tier / finalize work, steady per-tick
compute, and the per-request close-time lock stall — before picking a fix. The
2026-09-18 arms show a build-time VRAM shrink knob cannot move a host-tier /
pre-reserved-free-list cost: measure where the seconds actually are (the #732
sub-segments) before changing the allocator or the pool.
