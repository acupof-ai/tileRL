---
question: Does `tilerl train` score its own before/after eval at the rollout cap?
source: cpu (this Mac), tiny model; the 27B numbers it invalidates are in the linked entries
---

# The eval cap was documented, then shipped anyway for a day

## Context

[2026-09-04](2026-09-04-the-eval-cap-measured-itself.md) established that a GRPO
run's `gsm8k_before` of 39.0% was measuring the 256-token rollout cap rather than
the policy: mean completion 238.7 against a 256 cap, ~82.5% when scored uncapped.
The entry was written, the headline was withdrawn, and **the code was not
changed.** The live P1 run hit it again today.

## Root cause

`cmd_train` built one `SamplingParams` and used it for both jobs:

```python
params = sampling(tok, thinking, args.max_new_tokens, ...)   # cli.py:234
...
c, n, ntok = gsm8k_accuracy(engine, tok, eval_rows, params, ...)   # cli.py:297
```

`--max-new-tokens` is the **rollout** length — it is the training hyperparameter
the thinking-cap result is *about*. Passing it to the eval makes the before-arm,
the after-arm and the rollouts share one cap, so the eval cannot see any behaviour
longer than training allowed. `gsm8k_accuracy` already forced `temperature=0`, so
greedy was right and only the length leaked.

`eval.py`'s own docstring asserted the opposite — "the only place the cap is
absent" — which is how the bug survived a day of reading the code.

## Fix

A separate `eval_params` from a new `--eval-max-new-tokens` (default 2048, the
protocol the published before/after numbers were scored under), set explicitly in
both 27B recipes so it cannot silently follow `max_new_tokens`.

Gate: `test_the_eval_is_not_scored_at_the_rollout_cap` captures the
`max_new_tokens` that `gsm8k_accuracy` is **handed**. Asserting the flag parsed
would pass without the wiring; asserting on completion lengths would need a model
long-winded enough to reach the cap, and the tiny model is not. Negative control:
restoring `params` at the call site fails with `the eval ran at
max_new_tokens=4, the ROLLOUT cap`.

## `--load-adapter`, and why it is here

The running P1 saves `adapter.safetensors` before its after-eval, so its
after-number is recoverable by re-scoring the saved adapter uncapped — but nothing
in the tree loaded one back. (I declined to write this loader earlier as
speculative; it stopped being speculative when a real run's number needed
recovering.)

Two failure modes, both with gates:

- **Rebinding instead of copying.** The forward reads the objects `add_lora` put
  in `model.params`, so `trainable[k] = v` loads an adapter the model never sees
  and silently re-scores the base. `_load_adapter` uses `copy_`. Gate: greedy-decode
  the same prompt with and without the adapter and require the tokens to differ.
  Negative control — swap `copy_` for a rebind — fails.
- **Silently dropping keys.** An adapter saved before
  [#98](2026-09-05-half-the-lora-adapters-trained-nothing.md) carries
  `<weight>.scale.lora_*` and `conv1d.lora_*` keys that no longer exist. Unknown
  *or* missing keys raise rather than partially applying.

The gate trains its adapter with the **dense** reward (no `--data`), not
exact-match. First attempt used exact-match on a tiny random model, which scores 0
on every rollout: every group ties, `group_advantages` returns zeros, `lora_b`
never leaves its zero init, and the adapter is an exact no-op — the test failed
for a reason that had nothing to do with the loader. The assertion now checks
`lora_b` specifically is nonzero, because `lora_a`'s random init makes an
any-tensor check pass on a no-op adapter.

## Rule

**A documented bug is not a fixed bug.** This one had a dated entry, a withdrawn
headline and a named mechanism for a full day, and it still scored a live 27B run,
because the entry recorded the finding and nothing changed the line. When an entry
withdraws a number, it either lands the fix in the same tranche or says in one
line what still ships broken.

Second, smaller: **a shared object is a shared decision.** One `SamplingParams`
for rollouts and eval reads as economical and quietly ties two things that must
differ. The eval's length is not the training's length.
