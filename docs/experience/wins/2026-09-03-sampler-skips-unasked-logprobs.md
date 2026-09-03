# The sampler stops paying for scores nobody asked for — 2026-09-03

> Status: **counted, not timed.** `pending-remote` for wall clock. CPU suite
> 197 passed / 7 skipped, ruff clean.

## Context

Sampling and commit run on the host every tick, outside the captured decode
graph, so at B=1 they are the part of the 11.0 ms tick that does not shrink when
the kernels get faster. Three things in that path cost more than they had to.

**`sample_batch` always computed log probabilities.** `SamplingParams.logprobs`
defaults to False and the engine already discarded the result when no request
asked (`engine.py:816`), but the score was computed regardless — and for a greedy
row that score is `torch.log_softmax(logits, dim=-1).max(-1)`, **a second full
pass over 248320 logits**. Greedy is what the GSM8K and MMLU runs use, so every
tick of every eval paid for a number that was thrown away.

**`_restrict` ran once per row.** `torch.stack([_restrict(l, r.params) for ...])`
built the batch out of N separately-restricted rows. Each `_restrict` runs its own
`torch.topk` over the vocabulary, and when `allowed_ids` is set it also does
`torch.tensor(params.allowed_ids, device=...)` — a fresh host-to-device upload,
per row, per tick. The MMLU harness sets `allowed_ids`.

**`is_full_attn` rebuilt a frozenset per call.** `full_attn_layer_set` is a
`@property`, so `layer_idx in self.full_attn_layer_set` constructed a 16-element
frozenset on each of the 64 calls a forward makes.

## What Worked

| per tick | B=1 | B=8 | B=8 W=8 (verify) |
|---|---|---|---|
| full-vocabulary `log_softmax` for a discarded score | 1 → **0** | 1 → **0** | 1 → **0** |
| `topk` over the vocabulary in `_restrict` | 1 → 1 | 8 → **1** | 64 → **1** |
| `allowed_ids` host-to-device uploads | 1 → 1 | 8 → **1** | 64 → **1** |
| frozenset constructions per forward | 64 → **0** | 64 → **0** | 64 → **0** |

`sample_batch` takes `logprobs: bool` and returns `None` for the scores when it
is False; the engine passes `any(p.logprobs for p in params)`, which it was
already computing one line later.

`_sample_batch` stacks once and then restricts the batch. The restriction is
per-row in principle, so it checks first: when every row carries the same
`(allowed_ids, top_k)` — the case for every eval and every rollout — `_restrict`
runs once on `[N, V]`. `_restrict` was already written against the trailing
dimension, so this needed no change to it. Rows with differing cuts fall back to
the per-row path, which is no worse than before.

**Only the first row of that table is a B=1 win.** The restriction batching does
nothing at B=1, and `is_full_attn` runs at graph-capture time on the CUDA decode
path, not per tick. B=1 is the target — it is what a rollout runs at — so the
honest summary is: one full-vocabulary softmax per tick, removed.

## The tests

`test_sample_batch_matches_per_row` gained a check that `logprobs=False` returns
`None` and **draws the same tokens**. A sampler whose draw depends on whether the
caller wanted scores is the silent version of this bug.

`test_restriction_is_the_same_batched_as_per_row` compares batched `_restrict`
against per-row for top-k, allowed-ids, and both together. The failure it guards:
a batched `topk` that took the kth value across the batch rather than per row
would widen some rows' support and narrow others, and still sample without error.

## Bench snapshot

| date | commit | host | target | model | tok/s | notes |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this branch) | — | cpu | tiny | — | correctness only |
| pending-remote | | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | pending | B=1 d512 against 92.4; greedy, logprobs off |

## Rule

A default-off feature that is computed unconditionally is not free because the
flag is off. Check where the flag is read against where the work is done.
