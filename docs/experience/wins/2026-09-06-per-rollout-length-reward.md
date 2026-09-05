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
appends the new rows to `runs/<id>/rollouts.jsonl` **at the end of every step**,
beside the existing `eval-{before,after}.jsonl`.

Per step rather than once at the end, because the run this reacts to did not end
at the end. Nothing in `src/tilerl/` installs a signal handler, an `atexit` hook
or a `try/finally` around the loop; run 2 took a SIGTERM at step 45 and never
reached any writer — `eval-before.jsonl` survived only because it had already
been written. A single write after the loop would therefore have lost every row
of exactly the run that motivated the file. Appending per step costs nothing to
measure against: a row is 70 bytes, so a 100-step run at group 8 writes 55 KB in
100 appends. A killed run keeps every completed step's rows and loses at most the
step in flight — but only the rows. `length_reward_r` is computed after the loop
and reaches the manifest through `_finish`, so a SIGTERM'd run has the data and
not the number; recompute it from `rollouts.jsonl`. A run stopped by the drift
guard exits through `_finish` and reports it normally.

**A probe found that understated it: the manifest did not exist at all.** Both of
us reasoned from the code that a killed run loses `length_reward_r`. Signalling a
real training process by pid says worse — `write_manifest` is only reached inside
`_finish`, so a SIGTERM at step 6 left **12 rows and no `manifest.json`**, and
`tilerl ledger --json` printed `[]` with rc 0. `list_runs` globs
`*/manifest.json`, so the run was not in the ledger at all, and `rollouts.jsonl`
carries `step/g/tokens/reward/advantage` and no model, cap, group, lr, seed or
commit — the run id is a hash of those and cannot be inverted. The rows survived
and nothing could say which run made them.

One `write_manifest` before the loop fixes it; `_finish` overwrites it with the
finished manifest. Re-probed: `list_runs` **0 → 1**, and the manifest on disk
carries `model`, `commit`, `group`, `max_new_tokens` and the rest of the inputs.
The gate samples `len(list_runs(root))` at the start of every step and asserts it
is never 0; removing the pre-loop write turns it red with `list_runs saw
[0, 0, 0, 0, 0, 0, 0, 0, 0]`.

**Sending the signal a second time found what the fix leaves behind.** A cpu run
at group 8, SIGTERM by verified pid after 22 steps: `rollouts.jsonl` held exactly
`22 × 8 = 176` rows, the last one `step 22`, matching the last step line — the
per-step append loses nothing when the append is the last thing a step does. The
manifest is now on disk, `finished: null`, and `tilerl ledger` shows the run.

But it prints **`skip`**, rc 0:

```
c13cfab9f89b  train  running  skip
```

`_finish` never ran, so `gates` holds only the pre-seeded `rollouts_within_cap`
with `skipped: true`, and `format_run`'s verdict is `skip` when every gate is
skipped and `pass` when the list is empty — a run killed before step 1 reads
`pass`. Neither says "interrupted". The `running` in the timestamp column is the
only signal, and it comes from `finished: null` rather than from any gate. So the
ledger can now *see* an interrupted run, which is the point of the fix, but it
does not *judge* it: do not read a verdict off a run whose finished field is null.
Distinguishing "killed" from "passed" is a separate change and is not in this PR.

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

**And send the signal.** Two sessions read the same code and agreed that a killed
run loses one metric. It loses the whole manifest, and the ledger cannot see the
run at all. Reading a code path tells you what it does when reached; only killing
a real process tells you what is on disk when it is not.

Signalling once found the missing manifest; signalling again after the fix found
that the recovered manifest reads `skip` — one probe answers one question, and
the fix's own output is the next thing to probe.

## Results

`340 passed, 14 skipped` on cpu, ruff clean. Run 3's `length_reward_r` is the
first real value; the prediction the length term makes is a negative
within-group r that shrinks over training.
