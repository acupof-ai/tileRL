# Per-rollout (length, reward) logging, and the confound it removes — cpu, 2026-09-06

> Status: pending-remote (run 3 produces the first real numbers)

## Context

[The length-term bound](2026-09-06-what-a-length-term-can-recover.md) found that
run 2's mechanism claim — short rollouts score better, so the policy lengthens —
is stated on the wrong axis. GRPO's advantage is computed **within a group, on
one prompt**; run 2's evidence is a **cross-step** correlation over 45 steps that
each drew a different prompt, so prompt difficulty produces both longer
completions and lower reward and nothing separates them.

That is not a write-up problem. `grpo_loop` yielded
`float(np.mean(rewards))` and `float(np.mean([len(c) for c in comps]))`, so both
quantities were averaged over the group before anything recorded them. The
pairing never reached disk, and no re-analysis of run 2 can recover it.

## What Worked

`grpo_loop` takes `per_rollout: list | None`, extended with one dict per
completion — `step`, `g`, `tokens`, `reward`, `advantage` — in the same
out-parameter idiom `gsm8k_accuracy(per_problem=...)` already uses. The CLI
writes `runs/<id>/rollouts.jsonl` beside the existing `eval-{before,after}.jsonl`,
**including when the drift guard breaks the loop**: the rows up to the break are
the ones that describe the collapse.

`manifest["metrics"]["length_reward_r"]` is the Pearson r of (tokens, reward)
pooled over **within-group deviations**. Centering per group is the whole
mechanism: a hard prompt shifts both its lengths and its rewards, and that shift
is the confound.

**Measured on a fixture built to carry the confound and nothing else** — hard
prompts long and low-reward, easy ones short and high, but within each group the
reward varies with no relation to length:

| view | r |
|---|---:|
| pooled across steps (run 2's view) | **−0.8959** |
| centered within group (this metric) | **+0.0000** |

Same rows. The pooled number is entirely prompt difficulty. Both sign controls
still register a real within-group effect: longer-always-wrong gives **−0.9922**,
longer-always-right **+0.9975**, so the metric is not simply insensitive.

`None` rather than `0.0` when there is no variance to correlate — a tied group
contributes zero deviation in reward and cannot move r, which is correct because
it carries no signal. 19 of run 2's 45 steps were tied but not all of them, so
run 3 reports a number.

## Controls

Four mutations, each red on its own assertion:

| mutation | result |
|---|---|
| drop the per-group centering | r = **+0.484**, "prompt difficulty survived centering" |
| return `0.0` instead of `None` with no variance | `assert 0.0 is None` |
| record group means instead of per-rollout rows (the run-2 defect) | "2 rows for 2x2 rollouts" |
| reverse `adv` so each row gets another row's advantage | `[-1.0, 1.0]` against `[1., -1.]` |

The third is the defect itself reintroduced, and the gate names the row count.

**The fourth control passed on the first attempt, and that was the fixture's
fault.** The row count, `step` and `g` all survive a misaligned `zip`, so the
advantage had to be checked against `group_advantages` of the row's own reward.
With `reward_fn = len(c)` that check compares zeros: tiny never emits EOS, so
every completion is exactly `max_new_tokens`, the group ties, and
`group_advantages` returns all zeros — reversing zeros changes nothing. Keying
the reward on `c[0]`, which the per-rollout seed varies, makes the control fail.
The test now also asserts that at least one group did **not** tie, so the same
inertness cannot return silently: a fixture that stops producing reward variance
fails loudly instead of passing vacuously.

## Rule

When a claim is about a within-group relation, the per-row pairing has to survive
to disk — an aggregate computed at the source cannot be un-averaged later, and
the claim will get restated from the only axis the data supports. Report the
centered statistic next to the pooled one when both exist; their disagreement is
the finding.

## Results

`340 passed, 14 skipped` on cpu, ruff clean. Run 3's `length_reward_r` is the
first real value; the prediction the length term makes is a negative
within-group r that shrinks over training.
