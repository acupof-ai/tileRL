# TP training on real cards: what to measure first — approach, not code

No code, no PR. Every number below is either read off a run or marked unmeasured.

## The brief was wrong in three places, and each changes the order of work

**1. NCCL has already run on this pod, at world=6.** `/work/nccl6b.log`, dated
Aug 29, is a completed `torchrun --nproc_per_node=6 scripts/nccl_probe.py`:

```
world 6
       bytes  us/allreduce     GB/s  x64 layers ms
    10485760         124.5     84.2          15.94
     1310720          25.6     51.2           3.27
      163840         108.0      1.5          13.82
       20480          23.0      0.9           2.94
```

Four such logs exist (`nccl.log`, `nccl6.log`, `nccl6b.log`, `nccl6c.log`).
`errors/2026-09-06-the-nccl-floor-has-no-instrument.md` says the probe "is cited by
nothing; no run of it is recorded anywhere" — true of the tree, false of the pod. The
measurement predates the entry by a week and was never pulled back. That entry should
be closed against these numbers, not by a new run.

They mostly vindicate the floor the design pages consume: **23.0 µs at 20 KB and
25.6 µs at 1.3 MB** bracket the cited 21.5 µs, and the two ends do support "flat 20 KB
→ 1.3 MB". But the **163840-byte row reads 108.0 µs, 4.7x the rows on either side** —
non-monotonic, so either tenant contention (cards 2-7 were busy) or a real NCCL
protocol crossover sitting exactly inside the region three cost tables call flat. That
one row is the open question, not the floor.

**2. `tests/tp_world2.py` collecting 0 pytest items is by design, not the defect it
was described as.** `.github/workflows/ci.yml:47-62` runs the nine `tests/*_world[0-9].py`
gates as scripts with `uv run python3`, floors the count at ≥9, and the comment says
the zero-collection is measured and intended. Verified here: 9 gates present, and
`tp_world2.py` exits 0 with `54 updated tensors compared: match` on both the plain and
Adafactor arms. **The CI-visible gate that fails when the collective is skipped already
exists and already runs.** What it does not do is run on NCCL.

**3. The single-card baseline is two points on one ladder, not a backward and a step.**
`backward_secs` **23.194 s** is the `--no-instrument` warm reading at C=128
(`wins/2026-09-07-fp4-backward-warpgroup.md:80`). **71.529 s is also a backward**
(`wins/2026-09-06-one-grpo-step-is-54-percent-backward.md:24`), measured at tree
`a16ff9c` when `_GDN_CHUNK` was 16; the step in that entry is **131.579 s**. The chunk
ladder ran 80.207 → 41.446 → 34.717 → 23.194, so those two figures are the same
quantity at two rungs. Pairing them as backward-and-step is wrong.

**There is no current single-card step to compare TP against.** The most recent step
figures (132.891 → 138.588,
`wins/2026-09-07-a-layer-wide-checkpoint-segment.md:135`) were measured at `c61c1aa`,
which has neither the fp4 warpgroup fix nor C=128 as an ancestor — so that step carries
a ~70 s backward, not 23.2. Subtracting gives ~83 s as an estimate, not a reading. The
TP arm must measure one warm single-card step on its own tree, in the same session,
before the two-card run.

Also: **there is no "#192 recipe" in tileRL.** `src/tilerl/recipes.py` has six named
entries, none numbered; `b192` is a 0.2B dense run in the separate `/work/aupai` project.
The 27B recipe to use is `grpo-gsm8k-27b` (`recipes.py:19-23`), at prompt 256 / gen 1024
/ group 8 / LoRA-16 / micro 1.

## The blocker, fixed in #266

`LOCAL_RANK` was read **nowhere** in `src/` or `packages/`, and `torch.cuda.set_device`
only in `scripts/nccl_probe.py:17` and `scripts/bench_tp.py:59` — never in framework
code. `Backend.__init__` bound `torch.cuda.current_device()`, which is device 0 for every
rank unless the launcher sets `CUDA_VISIBLE_DEVICES` per rank, so a
`torchrun --nproc_per_node=2` of the training path put both ranks on card 0. NCCL states
it itself: `Duplicate GPU detected : rank 1 and rank 0 both on CUDA device`.

Fixed at `backend.py:330` — `LOCAL_RANK` is read and `torch.cuda.set_device` called when
it is in range, before `self.device` is bound.

This also decides the nccl-vs-gloo selection, which is one line —
`comm = "nccl" if self.device.type == "cuda" else "gloo"` (`backend.py:177`) — and it
reads `self.device`, bound before `init_tp` runs. So the device fix was a prerequisite
for the backend selection being meaningful, not a separate task.


## What is actually missing

Read off the tree, not inferred:

| piece | state |
|---|---|
| tape collectives with backwards | shipped — `autograd.py:274-277` registers `all_reduce`, `all_gather`, `cp_gather`, `tp_fork` |
| `--tp` flag, mesh, group construction | shipped — `cli.py:1108`, `_shard` at `cli.py:77-108` |
| sharded CE, TP-global clip, dp mean, Adafactor shard reduce | shipped, each with its own gate and error entry |
| gloo/CPU world=2 equality gate | shipped and running in CI |
| **the nccl-vs-gloo selection** | shipped at `backend.py:177`, and gated since #266 — `tests/tp_backend_world2.py` asserts the comm matches the device at world=2, which `tests/tp_world2.py` cannot (`RefBackend` hardcodes gloo) |
| **gradient bucketing** | absent |
| **compute/comms overlap** | absent |
| **27B TP config** | absent — no recipe sets `tp>1` |
| **any test on >1 CUDA device** | ten gates run in CI, all `TILERL_TARGET=cpu`/gloo; `tp_backend_world2.py` carries a cuda-only device-index arm that skips below two visible cards and **passed on 0+6 on 2026-09-07** |


The gap is narrower than "nothing beyond gloo/CPU". The correctness half is done and
gated. What is missing is one NCCL run and the numbers.

## The order I propose, cheapest first

**Step 1 — the floor, DONE.** `torchrun --nproc_per_node=2 scripts/nccl_probe.py` on an
idle 0+1 pair, three repeats: **20.6 / 20.7 / 21.4 µs** at 20 KB, 21.4–22.0 at 1.3 MB,
53.8–55.0 at 10 MB. The cited 21.5 was right to 4%, "flat 20 KB → 1.3 MB" holds (64x the
bytes for 4% more time), and the 108 µs outlier was contention — the same size reads
22.4 / 22.6 / 22.9 in the sibling Aug 29 runs. No table needs revising; the nine prose
citations are re-pointed at the measurement, and
`errors/2026-09-06-the-nccl-floor-has-no-instrument.md` is closed. Details in
[the floor was measured a week
early](experience/wins/2026-09-07-the-nccl-floor-was-measured-a-week-early.md).

The one consequence for what follows: at 20.6 µs a small collective is **pure latency**,
so the lever is call count, not message size.

Do this before any training arm. It is the only step whose output changes decisions
already made.

**Step 2 — the tiny-model world=2 step over NCCL, DONE.** Not `tests/tp_world2.py`'s body
as planned: that file drives `RefBackend`, which hardcodes gloo, so it cannot assert the
comm selection at all. `tests/tp_backend_world2.py` was written against the production
`Backend` instead, and passed on cards 0+6 at `8e60de3`:

```
production Backend world=2 on cuda/nccl: all_reduce [3.0], tp_fork bwd [3.0, 3.0],
sharded CE loss 2.161235 vs unsharded 2.161235
```

Of the two failure modes named in advance, the first happened — to the test rather than
to framework code. The gate's own probes were host tensors, and nccl raised
`No backend type associated with device type cpu`; every probe now builds on
`backend.device`. The `new_group` deadlock did not occur.

**Step 3 — the 27B TP=2 arm, and its own single-card control in the same session, DONE.**
`grpo-gsm8k-27b` at `group=8, micro=1, lora_rank=16`, identical seed and data on both
arms. `scripts/tp_step_arms.py` runs all four arms (tp=1 and tp=2, each instrumented and
bare) in one session against one content sha, and refuses when the TP arm times zero
collectives.

**TP=2 is 0.95x the single-card step** — 10.038 s against 10.596 s, per-card peak 43.97 →
25.46 GiB. Collectives are 5.95% of the step as an upper bound, and both of the obvious
readings of that number are wrong: `all_reduce`'s backward communicates nothing, so 99%
of it is `tp_fork`; and the two ranks differ 3x on `tp_fork`, so part of the figure is
rank skew absorbed by the collective. Bucketing and overlap are not scheduled off this.
Details, including the 456 s first-step JIT, in
[the entry](experience/wins/2026-09-07-tp2-on-two-cards.md).

## The profiler needed one change before it could see any of this — DONE in #264

`scripts/prof_backward_ops.py:517` called `_build_model(a.model, seed=0, keep_master=False)`
with no `tp=`, and `_shard` returns early at `tp <= 1` (`cli.py:86`). Run it under
torchrun as-is and you get two processes each doing an identical unsharded step with no
collective at all — a green run that measures nothing. `tp` and `backend` are now
threaded through, behind a `--tp` flag, and `tp_step_arms.py` refuses the whole
comparison when the TP arm times zero collectives.

Two things it still will not see, which the entry must state rather than leave as a
silent zero:

- **The optimizer-side reduce is outside the measured region.** `train.py:174` assigns
  `optimizer.tp_reduce = backend.all_reduce`, and `backward_secs` subtracts
  `optimizer_secs` (`train.py:320`). TP's optimizer collective lands in neither
  `backward_secs` nor any table row.
- **Collective rows are sync-serialized.** `instrument()` syncs before and after every
  handler (`:106`, `:110`), which removes whatever overlap the shipped path gets. The
  row is exposed latency plus rank skew, an upper bound, not the shipped cost.
  `--no-instrument` is the honest control.

## Which of the three unestablished pieces to do first

**Bucketing and overlap are both premature, and the probe says why.** On the idle pair,
20 KB costs 20.6 µs and 1.3 MB costs 21.4 — **64x the bytes for 4% more time**. A small
collective there is pure latency, so bucketing (fewer, larger calls) is exactly the right
lever *if* call count is the cost. But TP=2's call count and its share of the step are
both unmeasured: the design page's "128 all-reduces per tick ≈ 2.8 ms" was arithmetic
over the floor, and even now that the floor is real (128 × 20.6 µs = 2.64 ms) that is a
*decode* tick, not the training step this row is about. **Measure the step first, then
pick.** If collectives are 2% of a 23 s backward, neither bucketing nor overlap is worth
writing.

So: **the 27B TP=2 step time, and the collectives' share of it, is the first number.**
It is also the one that decides between the other two.

## Launcher

`pod_run.sh` takes one scalar card. Three lines assume it: `CUDA_VISIBLE_DEVICES=$CARD`
(`:70`), the orphan pre-check `nvidia-smi -i $CARD` (`:74`, whose `[ "$used" -gt 64 ]`
breaks on two lines of output), and the exit reading (`:130`). The claim helper already
takes a list — `card_claim.py status` currently shows `cards 2,3,4,5,6,7` on one claim —
so widening `pod_run.sh` is a small patch, not a mechanism. `pod_fan.sh` is the wrong
shape (N independent one-card jobs, no rendezvous, no claim) and should not be bent to
it. Precedent for the launch itself is `/work/nccl6b.sh`: `CUDA_VISIBLE_DEVICES` plus
`torchrun --master_port=<free>`.

**Every pair on this box is NVLink, not PCIe** — `nvidia-smi topo -m` reports `NV18` for
every GPU pair. Cards 0+1 share NUMA node 0; **the pair actually used is 0+6, which
straddles NUMA nodes** (CPU affinity 0-89 / node 0 against 90-179 / node 1). The
GPU-to-GPU path is NV18 either way, so the NUMA split affects host-side staging, not the
collective. Cards 1-5 and 7 are aupai's by the 09-05 ruling; 0 and 6 are tileRL's.


## Gates and entry

- **Correctness on NCCL**: `tests/tp_world2.py`'s body at world=2 over nccl. Its three
  negative controls come along — `--no-fork` deletes the backward collective and must
  fail. It cannot join the CI loop (that loop is CPU/gloo and CI has no cards), so it
  runs on the pod and its output goes in the entry.
- **The CI-visible gate already exists** and needs no work: ten gates, floor of 10,
  script-run so the 0-item collection is irrelevant. Worth one line in the entry saying
  so, because it has now been reported as broken twice.
- **Entry**: `docs/experience/wins/` with the world=2 probe table, the TP=2 vs
  single-card step on the same tree in the same session, and the collectives' share
  from the per-op table. Plus a correction closing
  `errors/2026-09-06-the-nccl-floor-has-no-instrument.md` against `/work/nccl6b.log`.

## What this rests on that is not measured

The floor is settled, so section (d2) of `design-parallel.md` — the 7x
ring-vs-all-gather ratio, the 2408 µs, the ~58K crossover — now rests on a real operand.
What it still rests on that is not measured is the **exposed** cost: how long a layer
actually stalls with ring's hops overlapping block compute. That page says so itself
(`:191`, "on floor arithmetic, pending an exposed-cost measurement"), and no CP attention
kernel exists to bench it.

Unmeasured and load-bearing for the steps below: **TP=2's collective count per training
step, and its share of the step.** Everything in "which piece first" waits on it.

The largest known cost, and it does apply here: **TP forfeits the decode graph** until
`Backend.all_reduce` is capturable, and that graph is worth **2.16x on the RL step**
(73.62 → 34.09 s, n=10, `wins/2026-09-05-recapture-after-update.md`). The RL path runs
`decode_graph=True, recapture_graph=True` today (`AGENTS.md:169`), so a TP training arm
starts roughly 2x behind and has to win that back before it is a gain. On two cards the
shard is 2-way, which is the least favourable side of that trade.
