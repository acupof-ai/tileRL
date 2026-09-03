# W=8 greedy is not the unspeculated greedy, and that is arithmetic — cuda(H20), 2026-09-03

## Context

The split-KV NaN is fixed. On `e26af2d` (GPU 5, separate run) the W=8 arm
runs clean: 0 NaN row-ticks of 393, every residue reached, 39 wide row-ticks
at the `n % 64 ∈ [1,7]` geometry that used to fail. The 0/500 GSM8K score
that started this was measured on `perf/spec-graph`, one commit below the
guard — a branch, not a commit, which is how it survived a "fixed on main"
answer.

What is left is a quieter claim. `Engine._verify` said "an accepted token is
bit-identical to the unspeculated one". On 8 GSM8K questions at width 8, 4 of
8 completions differ from the unspeculated arm — not corrupted, coherent
alternative wordings, same score:

```
base  '**1. Calculate the total number of eggs available:**\nJanet's ducks lay 16 eggs per day...'
spec  '**1. Calculate the total number of eggs used personally:**\nJanet eats 3 eggs for breakfast...'
```

Two candidates: a near-tie flipped by two tile geometries disagreeing in the
last bits, or an acceptance/indexing error. The discriminator is the top-2
gap at the divergence.

## Root Cause

Arithmetic, and the rate is predicted, not merely plausible. 27B NVFP4 +
`model_mtp.safetensors`, `main` `e26af2d`, GPU 6, greedy, `--gsm8k-n 8
--width 8`, `scripts/probe_spec_divergence.py`, run twice with identical
per-position logits:

| | |
|---|---|
| committed token ≠ the verify tile's own argmax | **0 of 1766** sampled entries |
| arm-to-arm \|Δ top-1 logit\|, identical history, n=920 | median **0.153**, p90 0.411, max 3.06 |
| base arm's top-2 gap over those same 920 positions | median **10.55**, p10 1.47, min 0.026 |
| positions whose top-2 gap is below the median Δ | **5 of 920 = 0.54%** |

Four of those five near-ties are the four divergences:

| q | index | base top-2 gap | arm-to-arm Δ top-1 |
|---|---:|---:|---:|
| 0 | 35 | 7.43e-02 | 5.62e-02 |
| 2 | 27 | 5.98e-02 | 2.36e-01 |
| 3 | 30 | 1.24e-02 | 1.60e-01 |
| 7 | 26 | 2.71e-02 | 1.37e-01 |

Acceptance is exonerated: every committed token is the verify tick's own
argmax, at every position, including the four that diverge. The two arms
compute the same position through different kernels. `Backend._plan` picks
`bM = 1 if m == 1 else _snap_mma_tile(m, 128)`, so a W=1 tick runs every fp4
and fp8 projection down the GEMV path and a W=8 tick runs the same
projections down an MMA tile — a different reduction order in every linear of
every layer. `paged_attention_decode`'s M tile is `snap_mma(G·W)`, so it
differs too. Their top-1 logits come out ~0.15 apart, about 2 bf16 ulps at
logit magnitude 24. Wherever the top-2 gap is under that, the argmax flips
and the completions fork. Five near-ties in 920 tokens is exactly 4
mismatching completions out of 8.

The recurrent state is not the source: `LinearStatePool`'s dtype is
`precision.dtype("recurrent_state")`, f32 on sm90, and `select_step` copies
between f32 planes, so the W=1 pool round trip loses nothing that carrying
the column across 8 register steps keeps.

Same commit, same session, the throughput claim reproduces: 2751 drafted /
1336 accepted (48.6%), 8 completions in 76 decode forwards against the base
arm's 259 — 22.78 tokens per decode forward against 6.77.

## Fix

No code path changes; bit-identity was never available and chasing it would
mean giving the verify tick the W=1 tile geometry, which is the whole win.
What changes is the claim and the gate.

`_verify`'s docstring now states the guarantee that holds on every backend:
**a committed token is this tick's own draw from the trunk at that chain
position**, under the per-generated-index seed the unspeculated arm uses.
String equality with the unspeculated arm is a CPU-reference property on top
of that, not the guarantee.

`test_verify_commits_the_trunks_own_draw` pins two invariants that are
backend-independent, both with a negative control on the CPU target:

| invariant | negative control | what fires |
|---|---|---|
| the committed token is the verify tile's argmax | `_verify` reads chain position `j+1` | `committed a token the verify tick did not draw` |
| the adopted step plane was written by this tick | `keep_steps = width - 1` | `adopted step planes this tick never wrote` |

The second closes a standing lead. `LinearStatePool.alloc_slot` zeroes
`states` and `conv_windows` but **not** `step_states`/`step_windows`, which
`select_step` reads back — so a plane an earlier tick wrote is a previous
slot owner's recurrent state. That leak is not live: `keep_steps` is
`max_i len(chains[i])`, `n_ok ≤ len(chains[i]) - 1`, and both write paths
(`reference.gdn_forward`'s step list and `gdn_decode_fused`'s `if t < ks`,
which has no `SeqQLens` gate) fill all `keep_steps` planes for every row of
the tick. Mechanized: tracking every `state_scatter(steps=True)` write per
tick and pairing it against every `select_step` read gives **0 violations**
across slot reuse, concurrency, ragged widths and chains trimmed to 1 or 0.
Poisoning both planes with NaN at `free_slot` and again before every tick
leaves the tiny-model completions unchanged. The test is what keeps it that
way — the invariant is spread over three files and nothing else states it.

## Rule

Speculation's contract is "the trunk's own draw", not "the same bits". A
verify tick that reaches its speed by changing the tile geometry cannot also
promise the unspeculated bytes, and a test that asserts string equality is
asserting a property of the CPU reference's reduction order.

When two arms disagree on greedy output, the discriminator is the **gap**,
not the diff: the losing arm's top-2 margin against the arm-to-arm logit
difference, distributed over every position, not shown on one example. A
divergence at a 1e-2 margin under a 1.5e-1 perturbation is arithmetic; one at
a wide margin, or where the committed token is not the tile's own argmax, is
a bug. Both numbers come out of the same instrumented run.
