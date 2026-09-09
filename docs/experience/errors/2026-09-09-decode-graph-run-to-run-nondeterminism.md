# The decode graph makes training runs non-reproducible across processes

## Context

The seed-0 GRPO control run collapsed (96,96,95,96,86,68,0×15). The before-arm
eval cache was suspected: a cache hit skips the before-arm (clean engine), a
miss runs it (pre-warmed), and the two paths diverged in 17/24 rollout rows.
But clean-vs-clean — same sha, same seed, same card, no eval — also diverged in
16/24 rows. The before-arm was confounded: the run-to-run nondeterminism made
every A/B comparison unattributable, and the collapse could not be blamed on
the config change under test.

## Root Cause

The captured decode graph is the run-to-run nondeterminism source. With it on,
two same-config runs differ in 16/24 rollout rows; with it off
(`--deterministic`), 0/24 (replicated twice).

The forward kernels themselves are bitwise deterministic:

- A fixed-layout two-forward test (same process, same engine, same blocks)
  gives 0/8 diff, max abs diff 0.000e+00 (`scripts/fwd_determinism.py`).
- The attention reduces by logical position, not physical block number:
  `p = k*block_N + i; bidx = p // block_size; KCache[BlockTable[bb, bidx]]`
  (`kernels_attn.py`). Physical block placement does not change the
  accumulation order. The pool is born `torch.zeros`
  (`kv_cache.py`), so unwritten slots are finite 0, not NaN — the causal mask
  alone then suffices (`-inf + NaN = NaN` would not).
- The graph capture and replay are deterministic in-process: two captures in
  one process, state reset between replays, give 0/8 diff
  (`scripts/capture_determinism.py`).

The first two-capture test reported 5.965 / 8.367 max-abs diffs and looked
like capture nondeterminism. The same-graph control — replay one graph twice —
differed too (8.367), which gave the game away: the decode forward ADVANCES
the gated-delta recurrent state (a per-slot mutable tensor that does not
route through the block table), so replay 2 read a different state than
replay 1. The 5.965 was state advancement, not capture nondeterminism.
Resetting the state and re-running the prefill before every replay gave 0/8.
The mechanism (capture nondeterminism) reasoned smoothly from an unverified
premise ("both replays read the same inputs") — the recurrent state was not
in the input dump because it does not route through the block table. Third
time this shape has bitten this investigation.

So the divergence is cross-process and graph-specific. The leading candidate
was uninitialized memory in the graph's private pool: a kernel reading an
intermediate before fully writing it would pick up process-specific garbage.
The poison test killed this. `scripts/poison_pool_determinism.py` fills the
pool's free memory with 0xFF vs 0x00 inside the capture — the forward's
intermediates reuse the freed block (within-capture free+reuse is 1.0000;
cross-capture reuse is 0.0000, per `scripts/pool_reuse_probe.py`) — and
compares replay logits: 0/8 diff, max abs 0.000e+00. The forward reads no
uninitialized pool memory. The cross-process source is unlocated; it is not a
read-before-write in the decode forward. Investigation is stopped here — the
candidate mechanisms are exhausted (forward kernels bitwise deterministic,
in-process capture deterministic, uninitialized pool read ruled out) and
`--deterministic` is a working workaround.

The collapse is not explained by this. The graph is the proven source of
trajectory divergence (on → 16/24, off → 0/24), but the mechanism is
unlocated, so "the collapse was bad luck" stays a hypothesis, not a
hypothesis-with-mechanism. The 100-step collapse-config run with the graph
off is the experiment that answers it, and it is queued.

## Fix

`--deterministic` runs the training rollout eager (`decode_graph=False`),
making A/B runs bitwise reproducible. The graph stays the default — eager
decode is ~6x slower on the rollout (14.7 vs 94.6 tok/s), so the flag is an
A/B switch, not a default flip. The graph's own cross-process nondeterminism is
unfixed.

## Rule

Before attributing a trajectory difference to a config change, prove the
config is the only difference: run the same config twice. A same-config pair
that diverges makes every A/B comparison unattributable until the
nondeterminism is located. And: a captured graph is not automatically a
deterministic function of its inputs — verify in-process before blaming
cross-process, and reset every mutable input (the gated-delta recurrent state
does not route through the block table) before replaying.
