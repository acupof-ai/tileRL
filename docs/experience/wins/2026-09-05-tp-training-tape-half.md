# TP training: the tape half, and four defects a gate found that reading did not, 2026-09-05

## Context

`tensor_parallel.py` has shipped weight sharding, alignment checks and GQA
replication since 08-30, and `Backend` has had `all_reduce`/`all_gather` for as
long. None of it trained: `_BWD` registered no collective, so every shipped
collective was inference-only. This is the backward half — three tape entries and
the gate that proves them.

## What worked

**The gate is the deliverable, not the code.** Two gloo ranks on the CPU target,
tiny model, comparing **all 54 gradient tensors** — sharded ones included — by
shipping the world=1 gradients through `shard_params` and requiring rank r's
gradient to equal slice r. `tests/tp_world2.py`, and it runs on a GPU-less
machine.

Comparing only the nine replicated params is the version that suggests itself,
and it would pass with every sharded gradient wrong. The tolerance is rtol 2e-3
against a measured bf16 spread of 1.4e-4; a missing collective is a factor of
`world`, four orders clear.

`--no-fork` deletes the backward collective and the gate must fail:

```
$ python3 tests/tp_world2.py            -> compared 54 tensors; TP-2 matches TP-1
$ python3 tests/tp_world2.py --no-fork  -> negative control: correctly FAILED
```

**Three entries, not the design's two.** `all_reduce` (identity backward),
`all_gather` (backward keeps this rank's chunk), and `tp_fork` — identity
forward, all-reduce backward — which `docs/design-parallel.md` did not name. It
is the dual of `all_reduce` and it is where the column-parallel gradient sum
lives.

## The four defects, and why none was findable by reading

**1. `all_reduce` returned its own input.** The tape addresses tensors by `id()`,
so backward wrote the gradient under `id(output)` and then popped it as consumed;
the input's gradient was simply gone. No exception, no NaN — `dX` came back
`None`. Fixed with `x.view_as(x)`, and `Tape.record` now refuses any op whose
output is identical to an input, because this is a tape-wide hazard that any
future in-place op inherits.

**2. The tied head must not fork.** `tiny` has `tie_word_embeddings=True`, so the
head is the replicated embedding table, not a vocab-parallel one: every rank
computes the whole dX there. Forking all-reduced it to 2x and scaled everything
below it — 54 of 54 tensors wrong, at sum-ratio exactly **2.0000**.

**3. `q_norm`/`k_norm`/`gdn_norm` are replicated weights applied to sharded
heads.** Each rank runs them over its own heads, so each rank's gradient is a
partial sum. Confirmed directly rather than argued: rank0 + rank1 == the world=1
gradient, exactly. They fork too — the pattern is about a replicated tensor with
sharded consumers, which covers weights as well as activations.

**4. `mp.Manager` deadlocks on torch tensors.** The first gate hung at 0% CPU on
every rank for four minutes. A two-rank probe returned in 3 s with `.tolist()`
and hung past 60 s with a tensor. Gradients now cross the process boundary as
lists. Worth knowing before writing any multi-rank harness.

Defects 2 and 3 both presented as a clean sum-ratio — 2.0000 and a partial —
which is what a gradient-count or loss-only check misses entirely. The loss was
correct in every one of these runs.

## What this does NOT establish

No 27B number: this is `pending-remote`. The mesh and the sharded cross-entropy
are not written. The CE is the substantive piece left — today's needs the full
vocab row and the vocab is sharded, so it has to be local max plus local sum-exp
combined by two scalar all-reduces, never materializing the 8.5 GiB f32 logits
that failed on 08-30.

TP still forfeits the decode graph, which is worth 2.16x on the RL step
([recapture-after-update](2026-09-05-recapture-after-update.md)). TP-8 starts 2x
behind and has to win that back.

## The only CPU config hid half the code

`tiny()` sets `tie_word_embeddings=True`, so its head is the replicated embedding
table. Every branch guarded by `not cfg.tie_word_embeddings` — the vocab-parallel
head, its fork, and the sharded cross-entropy that follows it — was therefore
**unreachable from CI**, on the only config this machine can run. The first
version of this gate passed with 54 tensors and a control that failed, and still
covered none of the path the 27B takes.

The gate now runs both layouts: 54 tensors tied, 56 untied, and `--no-fork` fails
on both. Two of us hit this from opposite ends within a minute — a review of the
gate's coverage, and a new op that turned out unreachable — which is what a
config-level hole looks like rather than a test-level one.

**"Unreachable from CI" understated it, and this entry was wrong when written.**
The gate file drives `mp.spawn` and prints its own verdict, so it defines no
`test_` function and `pytest` collected **zero items** from it; CI ran only
`pytest`. So neither layout ran in CI — the whole gate, control included, ran when
someone typed it. #121 adds the step that invokes it. Every claim above about what
"the gate covers" describes a hand-run until then.

## Rule

**A gradient gate must compare the sharded tensors, not just the replicated
ones.** The convenient subset is the one that agrees for the wrong reasons. And
when a test is widened to admit a new consumer — `_tp_fork` into the narrow-norm
allowlist here — inject a real violation and watch it fail, or the widening is
indistinguishable from deleting the test.
