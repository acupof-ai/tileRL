# TP=2 on two H20 cards: 0.95x the single-card step — 2026-09-07

## Context

The TP training stack had been correct on gloo/CPU since 09-05 and had never executed on
a GPU. [The rank binding](2026-09-07-every-rank-bound-card-zero.md) was the blocker, fixed
in #266, but the fix could not be tested: `world=2` over NCCL needs two visible cards, and
one card produces NCCL's own `Duplicate GPU detected` before any assertion runs. This is
the two-card run, on cards 0 and 6, at `8e60de3`.

## What Worked

**The gate passes on real NCCL:**

```
production Backend world=2 on cuda/nccl: all_reduce [3.0], tp_fork bwd [3.0, 3.0],
sharded CE loss 2.161235 vs unsharded 2.161235
```

Three checks that no CPU run can make, each the point of a different piece of #266:

* **The comm is nccl.** `tests/tp_world2.py` drives `RefBackend`, which hardcodes gloo
  (`testing.py:60`), so the production `Backend`'s own selection — `comm = "nccl" if
  self.device.type == "cuda" else "gloo"` at `backend.py:177` — had never been exercised
  at world=2. The gate asserts the comm matches the device rather than assuming it.
* **Each rank bound its own card.** The `idx != r` check has no CPU path and had never run
  anywhere. It is the direct proof of the binding fix: the same code on one card gave
  `Duplicate GPU detected : rank 1 and rank 0 both on CUDA device` this morning.
* **The sharded CE dispatch is right through a real process group** — not the math, which
  `ce_sharded_world2.py` already gates, but `Backend.cross_entropy_loss_grad`'s
  `tp_rank * vloc` offset and its `group=self._tp_pg` closure.

The loss is bit-identical to the CPU/gloo run at 2.161235. Same seed and tiny synthetic
model, so that is expected rather than lucky; what it rules out is nccl perturbing the
arithmetic.

**And TP=2 costs nothing per step.** Four arms, one session, one tree:

```
arm        ranks  tp    step_s     bwd_s     opt_s  i-step_s  i-coll_s   coll   coll%
control        1   1    10.596    10.505     0.090    14.267     0.000      0   0.00%
tp2x2          2   2    10.038     9.948     0.091    12.529     0.745   2696   5.95%
# tp2x2/control step 0.95x, backward 0.95x
```

Two cards are marginally **faster** per step than one, while sharding every matmul and
adding 2696 collectives. Per-card peak drops from 43.97 to 25.46 GiB, 18.8 GiB resident
per rank — the property TP exists for, read from `nvidia-smi` on both cards rather than
inferred.

## Two corrections to the attribution, both load-bearing

Anyone reading `coll% = 5.95` as "collectives cost 6% of the step, go bucket them" would
be wrong twice.

**`all_reduce` reads 8 µs/call, below the 20.6 µs NCCL floor.** Not an instrument
artifact and not a dropped collective. `_all_reduce`'s backward handler
(`autograd.py:225-230`) is `yield 0, g`: a row-parallel forward computes `Y = Σ_r Y_r`, so
`dY_r = dY` already holds on every rank and the backward has nothing to communicate. The
8 µs is Python dispatch plus the profiler's two syncs, and the row exists only so the
tape's `id()` chain crosses the collective — without it the residual add below reads a
tensor no entry produced. **99% of the collective time is one op, `tp_fork`** (identity
forward, all-reduce backward): 1672 calls, 0.7371 s.

A figure below a hardware floor should always be chased. Here it resolved to a handler
that correctly does nothing, but a silently-skipped collective looks identical.

**The two ranks disagree on `tp_fork` by 3x** — 0.7371 s on rank 0 against 0.2402 s on
rank 1, over the same 1672 calls. An all-reduce makes the early rank wait for the late
one, so the difference is rank skew absorbed by the collective, not communication. The
0.745 s is therefore an upper bound twice over: once for the profiler's syncs removing
overlap, once because part of it is a wait. **Someone optimizing "the collective" would
be optimizing a wait.** Rank 1's 0.2402 s is 2.39% of the step, and that is the closer
estimate of what TP comms actually cost; the 0.4969 s gap cannot be communication,
because both ranks move the same bytes through the same collective.

**What the gap is remains open, and this run cannot close it.** Two readings fit equally:
the shard split is uneven, so the same rank is always late and the fix is rebalancing
rather than anything comms-related; or arrival alternates per call and the gap is
straggler jitter with nothing to rebalance. Distinguishing them needs per-call timings,
and `instrument()` keeps only the sum and the count (`prof_backward_ops.py:124-137`) —
there is no per-call series to inspect, so the question is not answerable from these
JSONs. Naming it rather than picking the flattering reading: "the shards are uneven" is a
finding, "there is jitter" is not, and I have no evidence for either.


**And bucketing the calls recovers almost nothing.** At 0.441 ms/call, `tp_fork` runs
**21.4x the 20.6 µs NCCL floor**, so its time is not launch latency: 1672 × 20.6 µs =
34.4 ms, which is 4.7% of the op and **0.34% of the step**. Fusing 1672 calls into a
handful therefore buys 0.34%, not 6%. The call count is not the lever here, which is the
opposite of the decode case where a small collective at the floor makes count the only
lever.

One reading to avoid: "99% of the time is in one op, so the other 2695 calls are noise"
conflates op-share with call-share. The 1024 `all_reduce` calls are noise; the other
**1672 calls are `tp_fork`**. It is one op carrying most of the calls, not one call
carrying the op.

The consequence for the roadmap: at 0.745 s of a 10.0 s step, with both bounds loose,
only 0.34% of it recoverable by fusion, and `linear_attn_chunk` alone at 2.87-3.22 s,
gradient bucketing and compute/comms overlap are not worth scheduling off this
measurement.


## The first TP run on a box pays a full JIT rebuild, per rank

**Step 1 was 456.19 s against step 2's 10.04 s.** Two causes stack: both ranks JIT
independently, so every kernel compiles twice with no sharing; and the shard shapes are
new, so `/work/tilelang_cache` — warm for months of single-card runs — misses on
essentially every kernel the TP arm needs. 500+ `begins to compile` lines before the first
step.

**The cost is the autotuner, not the cache miss.** `tilerl-48` measured the contrast the
same afternoon: their fp8 kernels are cold-cache too and compile in 3-6 s, because
`write_tokens_fp8` and its siblings keep the launch geometry of their bf16 twins and
differ only in operand dtype — there is nothing to re-search. A new shard width changes
the GEMM tile shapes the autotuner searches over, so the miss lands on kernels with a real
tuning space. A shape change costs minutes; a dtype change on identical geometry costs
seconds.


This invalidated my own estimate, which is the reason it is here: I priced the window at
4 × 8m33s from the tp=1 control, a **cache-warm** configuration. A per-arm cost measured
on the control arm is not a cost for the treatment arm when the treatment changes the
kernel shapes.

Warming the cache, or sharing one compile across ranks, is a prerequisite for any TP
wall-clock claim — not an optimization to consider afterwards.

## Which numbers survive, and which do not

Arm 4 ran against the cache arm 3 had just warmed, so no total-runtime figure across these
four arms means anything. Every arm is read at **step 2** — the warm step, after the JIT
is paid — which is cache-independent by construction.

The instrument costs 3.671 s of a 10.596 s control step (35%) and 2.491 s of the tp=2
step. That is why the bare arms are the wall clock and the instrumented arms are read only
for the per-op table and the collective count.

`opt_s` (0.090 / 0.091 s) is **derived** as `train_secs - backward_secs`, because
`rl_step` subtracts `optimizer_secs` from `backward_secs` (`train.py:320`) and the
profiler emits neither a key nor a table row for it — so TP's optimizer all-reduce
(`train.py:174`) appears nowhere else. On CUDA it also carries the profiler's trailing
`torch.cuda.synchronize`, making it an upper bound on the optimizer alone. It is
unchanged between the arms, which is the answer to whether TP's optimizer collective
costs anything measurable: it does not.

## Rule

A guard whose condition cannot occur on the test machine is not a guard yet. The
device-index check, the CPU-tensor bug and the duplicate-GPU refusal all sat behind
`device_count() >= 2`, and the gloo/CPU suite was green through every one of them.

The measurement counterpart: three of today's four defects were found on one card, and
only the topology could not be. Worth separating, because it says what a one-card window
is worth.

| defect | found on | why |
|---|---|---|
| every rank binds card 0 | one card | NCCL's own duplicate check states it |
| gate probes were host tensors | one card | nccl has no CPU backend, raises immediately |
| `pod_run.sh` backtick ran on the laptop | one card | stderr at assembly time, machine-independent |
| the comm selection and device-index checks | **two cards** | both are `device_count() >= 2` guarded |
| TP's actual step cost | **two cards** | there is no one-card version of it |

## Results

| date | commit | machine | target | what | result |
|---|---|---|---|---|---|
| 2026-09-07 | 8e60de3 | H20 cards 0+6 | cuda | world=2 gate over NCCL | pass, all three cuda-only checks |
| 2026-09-07 | 8e60de3 | this Mac | cpu | same gate + 2 controls | pass; both controls fail correctly |
| 2026-09-07 | 016ab33b | H20 card 0 | cuda | tp=1 bare, step 2 | **10.596 s** step, 10.505 bwd, 43.97 GiB |
| 2026-09-07 | 016ab33b | H20 card 0 | cuda | tp=1 instrumented, step 2 | 14.267 s step |
| 2026-09-07 | 016ab33b | H20 cards 0+6 | cuda | tp=2 bare, step 2 | **10.038 s** step, 9.948 bwd, 25.46 GiB |
| 2026-09-07 | 016ab33b | H20 cards 0+6 | cuda | tp=2 instrumented, step 2 | 12.529 s, 0.745 s coll, 2696 calls |
| 2026-09-07 | 016ab33b | H20 cards 0+6 | cuda | tp=2 step 1 (the JIT) | 456.19 s |

Settings: `qwen38-27b`, group=8, gen=256, micro=1, lora_rank=16, prompt 256, blocks 4096,
identical seed and data in every arm.

**Provenance is a content sha, not HEAD.** `016ab33b5f36` hashes the profiler plus
`src/tilerl/**.py` plus `tilerl_kernels/**.py`. The pod tree is not a git repo, so the
arm header printed `HEAD unknown`; the content sha is what actually identifies the bytes
that ran, and it was identical across all four arms — which is the check that no peer
edited the shared checkout mid-run. The tree corresponds to `8e60de3` plus this branch's
uncommitted state at run time.

## Known limitation of the harness

Each arm spawns a fresh process and pays a full 27B load. `tilerl-48`'s probe shares one
model object across two engine builds for exactly this reason, and doing the same here
would cut most of the wall clock. Recorded rather than presented as a floor:
`tp_step_arms.py` could load once and run all four arms in-process.

With one caveat that makes the naive version worse than the loads it saves, from 48's own
card run: the **engines** hold the memory, not the model. Two arms each holding a fitted
KV pool left the third sizing itself into what remained and asking for 47.50 GiB.
`empty_cache` frees nothing while an engine is still bound, and an `Engine` sits in
reference cycles so `del` alone does not release it — `gc.collect()` is required. Sharing
the model without releasing each arm's engine turns the saving into an OOM at arm 3.

