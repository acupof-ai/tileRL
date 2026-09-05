# Recapture after update: the decode graph is worth 2.16x on the 27B RL step

**Date:** 2026-09-05
**Source:** H20 sm90 card 6, Qwen3.8-27B-NVFP4, `grpo` group=8 `max_new_tokens=256`,
tilelang 0.1.13 (`/work/tl013`). Tree: branch `a4a9473` (#94's waiver split) plus
PR #98's code hunk, verified byte-identical to `bbc3169`'s diff. **Without**
v100's prefix-publish fix.

## Context

`grpo_loop` refused an engine carrying a decode graph or a live prefix store,
because both sample from an earlier policy after an optimizer step, silently.
[#94](https://github.com/acupof-ai/tileRL/pull/94) adds `invalidate_weights()` and
two waivers — `recapture_graph=` and `clear_prefix=` — so a training engine can
keep them and clear them per step. The question was what that buys.

## What worked

Four arms, differing only in how the engine is built and which waiver
`grpo_loop` gets. Six steps each, step 0 dropped (it pays capture), **run in both
orders** because the arms share one process:

| arm | forward | reversed | pooled n=10 | vs baseline |
|---|---:|---:|---:|---:|
| baseline (graphs off, `NoPrefixStore`) | 73.12 ± 0.90 | 74.12 ± 2.07 | 73.62 | — |
| prefix (`clear_prefix=True`) | 76.28 ± 0.39 | 73.38 ± 2.71 | 74.83 | +1.21 s, 0.98x |
| graph (`recapture_graph=True`) | 34.43 ± 0.30 | 33.74 ± 1.23 | **34.09** | **−39.53 s, 2.16x** |
| both | 34.67 ± 0.37 | 33.51 ± 0.13 | **34.09** | −39.53 s, 2.16x |

**A 100-step 27B GRPO run goes 123 min → 57 min.** Capture pays for itself inside
step 0, which ran 46.81 s including it.

`graph` and `both` pool to the same 34.09: the prefix store's cost is real with
graphs off and gone with them on. The store's cost is host-side bookkeeping, and a
captured graph removes host time from the decode tick, so work that sat in a launch
gap now overlaps GPU work — that reading is unproven, the two numbers are not.

## And it is not fast because it is stale

A graph replaying old weights would also be fast, so the speed number means nothing
without this. `decode_graph=True`, greedy, seed 0:

```
graphs_held_before_update     1      the state check is not vacuous: a graph captured
zero_perturbation_same     True      CONTROL: invalidate + re-rollout on UNCHANGED
                                     weights reproduces the rollout exactly
perturbation_changed_rollout True    a large perturbation to embed_tokens changes it
verdict                    PASS
```

Without the zero-perturbation control, "the output changed" is also what sampling
noise looks like.

## The prefix store publishes 739 entries and serves 0

`prefix_published: 739, prefix_hits: 0` in every prefix-carrying arm, in both
orders. Publish is gated on `materialized % BLOCK_TOKENS == 0` **and** decode
phase (`engine.py:1100`), so a prompt boundary never fires it and the 8 rollouts
of a group never share their prefill. Costs bookkeeping, returns nothing, until
v100's block-granular store lands.

## Two things this measurement got wrong first

**A number measured in one position is not a measurement.** The forward order gave
prefix = +3.16 s against baseline at t=7.2 with a 95% CI of +2.15…+4.18 — tight,
significant, and posted to the board. The reversed order gives 73.38 ± 2.71, and
the reversed arm *declines* monotonically across four steps (77.8, 71.8, 70.9,
72.5) rather than holding a level. Pooled, the effect is +1.21 s at t≈2. The
forward arm's sd of 0.39 is what made a position artifact look like a result. The
drift's cause is undiagnosed; the counters rule out eviction.

**On cpu the ordering effect is catastrophic and would have inverted the whole
table.** Whichever arm ran first was 400–1500× slower, a different arm each time:

```
forward   baseline [32.9, 40.1]   prefix [.075,.096]  graph [2.5,.093]  both [.074,.094]
reversed  both [140.7, 75.2]      graph [.075,.091]   prefix [2.6,.088] baseline [.073,.09]
```

That is the TileLang JIT compiling every shape into a cold cache, charged to arm
#1. Read in forward order alone it says "baseline costs 40 s/step and every waiver
is free" — clean, publishable, entirely false. The pod's warm 2.5 GB cache removes
it, which is why the card rows are flat and the cpu rows are not.

A third error never reached a number: the script called `add_lora` **before**
`build_engine`, and `build_engine` materializes the params, so the adapters pointed
at tensors the forward never reads and the tape produced no gradients. It failed
loudly on the first arm. `cli.py:286` already carried the comment saying the order
matters.

## Method: what the launcher did wrong

Every job here was started as
`crictl exec <cid> bash -lc '... nohup setsid python3 x.py > log 2>&1 & echo $!'`.
The wrapper shell exits immediately, which detaches the job — and orphans it to
container PID 1, which in this container is **`sleep infinity`**, a process that
never calls `wait()`. So all four job pids ended as `Zs` with ppid 1, permanently
unreaped. One of them held its 28.2 GB CUDA context long enough for another team
to see it on card 6 and consider a container restart, which would have killed the
P1 run on card 0.

The launcher wants a parent that reaps: run in the foreground under `crictl exec`
with a monitor tailing the log, or keep an outer shell alive on `wait`. (Neither
alternative is verified in this container, so that is advice, not a measured rule.)

**And three readers disagree about whether a process is dead.** `kill -0` returns
success for a zombie, `/proc/<pid>` still exists for a zombie, and `pgrep -f
<pattern>` matched this session's own `tail -f` and the `pgrep` command line
itself. Only `ps -o stat=` shows `Zs`. That cost a chained run that hung forever in
`while kill -0 <pid>; do sleep 10; done` against an already-dead target, and two
false "the job survived" readings. `card_claim.py` prints exactly this warning when
it refuses a zombie claim; it said so hours before I generalised it.

## Rule

**Run the arms in both orders when they share a process.** Any resource an arm
warms — a JIT cache, an allocator, a graph pool — is inherited by the arms after
it, so a single-order table measures position as much as treatment. The control
costs one extra run and it changed one conclusion here and would have inverted
four on cpu.

**And a speed number for a cache needs a staleness check beside it.** "Faster"
and "wrong" are the same measurement until something proves the cache resamples.
