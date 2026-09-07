# Every rank of a torchrun launch bound card 0 — H20, 2026-09-07

> Status: partial. The gloo/CPU half and the one-card NCCL findings are measured here;
> the two-card gate and the tp=2 arms are `pending-remote` at the bottom, with their
> exact commands.

## Context

The TP training stack has been correct on gloo/CPU since 09-05 — tape collectives with
backwards, sharded CE, TP-global clip, dp mean reduce, Adafactor shard reduce, each with
a gate. None of it had ever run on a GPU.

Two things blocked that, and neither was visible from the gloo side.

## What Worked

**`LOCAL_RANK` was read nowhere.** `grep -rn LOCAL_RANK src/ packages/` returned nothing,
and `torch.cuda.set_device` appeared only in two scripts, never in framework code.
`Backend.__init__` binds `torch.device("cuda", torch.cuda.current_device())`, which is
card 0 for every rank unless the launcher intervenes. So a `torchrun --nproc_per_node=2`
of the training path put both ranks on one card.

The fix is four lines, placed before `self.device` is set — which also has to be before
`init_tp`'s `comm = "nccl" if self.device.type == "cuda" else "gloo"`, since that reads
the device this binding chooses:

```python
local = int(os.environ.get("LOCAL_RANK", -1))
if 0 <= local < torch.cuda.device_count():
    torch.cuda.set_device(local)
```

The range test is the whole design. Out of range covers both launcher shapes: no
torchrun leaves `LOCAL_RANK` absent, and a launcher that already pinned one card per rank
leaves it past `device_count()`.

**NCCL confirmed the bug before the fix could be tested.** Running the world=2 gate on
card 0, NCCL refused with its own duplicate check rather than any assertion of ours:

```
ncclInvalidUsage: Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 65020
```

That is the defect stated by the library, on the unfixed path, and it is a better witness
than the gate's own assertion would have been.

**A gate on the production `Backend`.** `tests/tp_world2.py` drives `RefBackend`, which
hardcodes gloo (`testing.py:60`), so at world=2 nothing exercised the real comm selection,
the collectives through a real process group, or the `tp_world > 1` arm of
`cross_entropy_loss_grad`. `tests/tp_backend_world2.py` does. On CPU:

```
production Backend world=2 on cpu/gloo: all_reduce [3.0], tp_fork bwd [3.0, 3.0],
sharded CE loss 2.161235 vs unsharded 2.161235
```

Two controls, each failing on its own assertion: `--no-collective` makes `all_reduce` the
identity, `--rank0-shard` has every rank claim rank 0's vocabulary slice (loss 2.161 →
2.549). A third check has no CPU path at all — each rank must bind its own cuda index —
so it is written to fail by name on the first two-card run rather than mocked.

The CE hole is narrower than it first looked, and worth stating precisely:
`tests/ce_sharded_world2.py` already gates `reference.cross_entropy_sharded`, the math.
What had no cover is `Backend.cross_entropy_loss_grad`'s **dispatch** — the
`tp_rank * vloc` offset and the `group=self._tp_pg` closure. `--rank0-shard` is exactly
that offset bug.

## What the one-card window taught, which two cards would have hidden

**The gate's own probes were host tensors.** On CPU/gloo that is fine; gloo accepts them.
The first card-0 run died at `backend.all_reduce(probe)`:

```
RuntimeError: No backend type associated with device type cpu
```

This is the failure mode the approach note predicted in the abstract — "gloo accepts CPU
tensors and nccl does not" — landing on the test rather than on framework code. Every
probe now builds on `backend.device`.

**World=2 over NCCL is impossible on one card, by construction.** `pod_run.sh` exports
`CUDA_VISIBLE_DEVICES=0`, so `device_count()` is 1, rank 1's `LOCAL_RANK=1` falls out of
range, and the binding correctly declines to bind a card that is not there. There is no
one-card version of this measurement, the same way there is no one-card version of a
link benchmark. The gate now says so instead of dying inside NCCL:

```
SKIP: world=2 over nccl needs 2 visible cards, saw 1
```

**A backtick in a comment ran on the laptop.** Every `pod_run.sh` launch printed
`scripts/pod_run.sh: line 63: pod_run_claim: command not found`. `RUNNER_EOF` is
unquoted, so a backticked word in a runner comment is a command substitution evaluated at
assembly time on the calling machine. The emitted runner was still valid and the card was
still claimed, so the only symptom was a stderr line that read like a pod error.

The selftest could not have caught it: it captures the runner on stdout, which is correct
either way. The witness is stderr from the emit, and that is now arm 0. Checked against
the mutant — with the backtick restored, arm 0 fails with that exact line.

## Rule

A guard whose condition cannot occur on the machine that runs the tests is not a guard
yet. The device-index check, the CPU-tensor bug and the duplicate-GPU refusal all sat
behind `device_count() >= 2`, and the gloo/CPU suite was green through every one of them.
When a subsystem's whole point is more than one device, the CI-visible half proves the
logic and none of the topology, and the entry has to say which half it is reporting.

The backtick has the same shape from the other side: an error on the wrong machine's
stderr, in a channel no assertion read. Both were found by running the thing rather than
by testing it, which is the argument for the one-card window — it found two defects in
the test and one in the launcher, none of which a two-card run would have shown
separately from the measurement it was there to take.


## Results

| date | commit | machine | target | what | result |
|---|---|---|---|---|---|
| 2026-09-07 | 4dacbcf + gate edits | H20 card 0 | cuda | world=2 gate, 1 card | SKIP, by construction |
| 2026-09-07 | 8581262, pre-fix | H20 card 0 | cuda | unfixed binding, 2 ranks | NCCL "Duplicate GPU detected" |
| 2026-09-07 | 4dacbcf | H20 card 0 | cuda | 27B tp=1 bare, 2 steps | 22.09 s/step bwd, 62.62 GiB peak |
| 2026-09-07 | f0c8168 | this Mac | cpu | world=2 gate + 2 controls | pass; both controls fail correctly |
| 2026-09-07 | f0c8168 | this Mac | cpu | 10 distributed gates | 10/10, at the CI floor |
| 2026-09-07 | f0c8168 | this Mac | cpu | full suite | 468 passed, 15 skipped, 1 xfailed |

The commit column is uneven on purpose. `4dacbcf` is the sha the pod tree reported, but
the gate file carried edits past it when the SKIP was observed, and the rebuild of these
commits gave them new shas — so the cuda rows name what actually ran rather than a sha
that would imply a clean tree. The tp=1 arm is the one measurement here whose bytes are
pinned, and only because `prof_backward_ops.py` and its imports were unmodified at
`4dacbcf`.

The tp=1 row is the control the tp=2 arms are read against, and it is worth one line of
caution: it was taken in a *different pod session* from the arms that will follow, so it
is a reference point, not the baseline of the comparison. `tp_step_arms.py` exists
precisely because a cross-session pair is not a measurement — it runs both arms in one
session against one content sha. When the two cards arrive, the tp=1 arm is re-run there
and this number is only a sanity check on it, not a substitute.

At 22.09 s/step, the [20.6 µs](2026-09-07-the-nccl-floor-was-measured-a-week-early.md)
floor is 1 part in 10⁶ of a step. That does not make collectives free — the count is what
matters and it is unknown until the tp=2 arm runs — but it does bound the arithmetic: TP
would need on the order of 10⁵ collectives per step before the floor alone is the cost,
so anything smaller shows up as bandwidth or as a sync stall, not as launch latency.



## Pending-remote

Both need two cards. Cards 0 and 6 are tileRL's by the 09-05 ruling; 6 returns from
aupai's run later today, and cards 1-5,7 are aupai's and are not to be taken.

```
# the NCCL half of the gate: the comm assertion picks nccl, and the device-index
# check fires for the first time
scripts/pod_run.sh tpgate 0,6 -- python3 -u tests/tp_backend_world2.py

# the 27B TP=2 arms against a same-session tp=1 control
scripts/pod_run.sh tpstep 0,6 -- python3 -u scripts/tp_step_arms.py \
    --out-dir /work/tpstep --tp 2
```

The second is the one the row turns on: the collectives' share of a training step decides
whether gradient bucketing or compute/comms overlap is worth writing, and at a
[20.6 µs floor](2026-09-07-the-nccl-floor-was-measured-a-week-early.md) a small collective
is pure latency, so the lever would be call count rather than message size.
