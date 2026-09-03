---
question: Why did Engine.logprobs() return a number that no distribution in the engine produced?
status: measured
source: tileRL 3cdb3a9, found reading engine.py while scoping docs/rl-sota-parity.md §2
---

# The sampler and the scorer disagreed about which distribution drew the token

`Engine.logprobs()` promised "log p of each returned token" under "the
distribution it was drawn from" (`engine.py:420-432`). It returned log p under a
different distribution — one the engine never samples from.

Two functions applied two different truncation rules to the same logits:

- `_restrict` (`engine.py:87-96`) applies `allowed_ids` and `top_k`, and **not
  `top_p`**.
- `reference.sample_batch` (`reference.py:962-969`) applies `top_p` on top of
  that and renormalizes before drawing.

`_sample_batch` scored with `log_softmax(_restrict(logits)/T)`, so the reported
value omitted the `top_p` renormalization. The error is exactly
`-log(kept top_p mass)`: reported too low by that amount, on every token, in the
same direction.

## Measured

At the 27B's `V=248320` and the model card's non-thinking sampler
(`T=0.7, top_p=0.8, top_k=20`), over 8 synthetic rows:

| | |
|---|---|
| mean gap | +0.2121 nats |
| max gap | +0.2226 nats |
| mean kept top_p mass | 0.8089 |
| `gap == -log(kept mass)` | exact, 8/8 rows |

Tiny model, real decode path, 6 tokens: mean +0.0447 nats, max +0.1378. The
tiny model's distribution is sharp, which produces the cleanest evidence — at
`top_p=0.8` the nucleus is often a **single** token, so true log q is exactly
0.0 (the draw was deterministic) while the engine reported −0.054 and −0.138.
It was reporting uncertainty that did not exist.

Second finding, worth its own line: `top_p=0.8` cuts the nucleus to **9–14
tokens** on top of `top_k=20`. `top_k=20` is not the truncation width; the
nucleus boundary is, and it moves per position.

## Why the existing test passed

`test_logprobs_are_returned_and_deterministic` asserted one value per token,
none positive, and reproducibility under the same seed. All three hold for a
number that is wrong by a constant offset. The test never compared against an
independently computed log q, so it constrained the shape and not the value.
Confirmed by reverting the fix and watching the strengthened test fail
(`token 0: reported -0.054 vs log q 0.0`).

## Fix

One definition of the nucleus, `reference.top_p_probs`, used by `sample`,
`sample_batch`, and the test's oracle. `sample_batch` now returns
`(tokens, logprobs)` read off the same `probs` tensor it draws from, so the two
cannot diverge again — the scorer no longer recomputes anything.

Sampling behaviour is unchanged: the draw still happens in
descending-probability order via `multinomial` on the same tensor, so every
sampled token in the repo is bit-identical. `test_sample_batch_matches_per_row`
holds.

## Rule

**When two functions apply the same rule to the same data, one of them will
eventually apply a different rule.** A sampled token and its score are one
result, not two computations that happen to agree — return them together from
the code that owns the distribution.

And: a logprob test that checks sign, count and determinism has not checked the
logprob. Compare against an independently derived value, or the assertion
passes for any constant offset.

## Downstream

- `messages.py:172` and `server.py:190` return these numbers to clients. Both
  were wrong by the same offset until this fix.
- `docs/rl-sota-parity.md` §2 says of option B (importance weighting) that "we
  have the ingredients and do not use them". Half true: the ingredient was
  defective, so option B was not implementable before this landed.
- The spec-decode path is unaffected. Its acceptance accounting uses
  `DraftHead.confidence` off `backend.greedy` (`engine.py:769,804`), never
  `_last_logprobs`, so no published spec number moves.
