# Async RL's ceiling is 1.36x, and the reference implementation does not use it on its own 30B — 2026-09-08

**Status:** async rollout/training overlap is **rejected**. Read from
[OpenBMB/Meshy](https://github.com/OpenBMB/Meshy) at `tilerl-27`'s request; the decision is
theirs, the code references are mine.

## The prize

A GRPO step on the 27B (group 8, gen 1024, micro 1) is sequential:

| phase | secs | share |
|---|---:|---:|
| rollout | 63.156 | 73.77% |
| forward | 4.415 | |
| backward | 17.957 | |
| optimizer | 0.087 | |
| **sum** | **85.617** | |

Overlapping rollout with training turns the step from `sum` into
`max(rollout, train)` = 63.156 s. **1.356x, on every step, independent of step count.**

## What it costs, from Meshy's code

Meshy is an LLM RL framework built on SGLang and torchtitan (8 commits, 59 stars at reading;
so its README is a claim, not a track record). It implements exactly this overlap, which is
what makes it worth reading — not its architecture, which is multi-service SPMD against our
one card in one process.

**The staleness knob is one integer, and the "one code path for sync and async" claim holds.**
`meshy/config.py:227` declares `pacing_window: int | str | None = "auto"`, and
`meshy/worker/rollout.py:243-247` is the whole mechanism:

```python
def _generation_budget(self) -> int:
    if self.gates_seen == 0:
        return 0
    return (self.gates_seen - 1 + self.pacing_window) * self.train_batch_size
```

with one wait loop at `:293` (`while self.samples_started + num_samples >
self._generation_budget()`). `1` is lock-step, `2` permits a batch of lead, `None` is
unbounded. Searching `meshy/` for an async-mode branch returns nothing — the only
`mode == "async"` in the tree is `scripts/analyze_step_timing.py:581`, an analysis script.

**The algorithmic price is off-policy correction, which the framework pays.**
`meshy/backend/titan/trainer.py:23-27`:

> **Asymmetric PPO clip** with `ppo_clip_eps_low` and `ppo_clip_eps_high`.
> **Behavior policy** for the importance ratio is configurable via `old_logprobs_source`:
> `"rollout"` takes SGLang's per-token logprobs (already packed into the batch); `"train"`
> does an extra no-grad forward through the current training model over the same micro plan.

**Weight sync stops the rollout.** `recipe/justrl_async.py:13-15`: the trainer aborts
in-flight generation (`pause_generation(mode="abort")`), does the train/infer GPU hand-off
plus weight sync, then `continue_generation`; aborted rollouts are resubmitted by the driver.
So the answer to "does generation pause" is yes, with the interrupted work redone.

## The criterion that decided it

**A framework built for asynchronous RL runs its own largest production recipe
synchronously.** `recipe/justrl_qwen3_30b_a3b.py:15` documents its own configuration:

> **rollout**: 1 CPU-only rollout driver, lock-step (``pacing_window=1``).

Across 14 recipes, 11 pass `pacing_window=1`. Only `justrl_async.py` (2),
`justrl_fully_async.py` (None) and `math_grpo_minicpm5_2_6b.py` (None) are asynchronous.

The people with the mechanism, the tuning surface and the motive chose sync at 30B. That is
stronger evidence than any throughput argument we could construct, because it is a revealed
preference under exactly the constraints we would be adopting.

## What we would have to build first

Our `rl_step` computes an advantage and multiplies it in. There is **no importance ratio and
no clip anywhere in it** — we are on-policy by construction, which is also why `grpo_loop`
refuses an engine with the decode graph or the prefix store on. Async needs per-token old
logprobs, a ratio, and an asymmetric clip: a structural change to the gradient path, taken
on the same day a gradient defect (an empty rollout setting its group's baseline) was found
and fixed.

Notably the wall clock does not object to the safer variant: `old_logprobs_source="train"`
costs an extra no-grad forward, 4.415 s, and `max(63.156, 26.874)` is still 63.156. **The
price is entirely in gradient correctness, not in the 1.36x.** That makes the decision a
question about risk, not about speed.

## Two levers on one quantity do not multiply

Rollout is 73.77% of the step and is itself partly idle — slots that finished early while the
group's longest row runs on. **The size of that idle fraction is currently unmeasured**: the
probe that produced the figure we had been using did not render its prompts through
`render_chat`, so the model received raw documents and wrote to the cap, and every number
derived from that length distribution is void.

The ordering argument does not need the number, only that idle > 0. If idle is fixed first,
rollout shrinks and `max(rollout, train)` saves the same seconds while the *ratio* rises; if
async lands first, part of the idle fix's benefit is absorbed by the `max`. **Sequential
levers on one quantity give a sum that depends on their order, so a priority list built from
independently-computed ratios overstates the total.**

## Rules

- **Ask what a reference implementation runs in production, not what it supports.** A
  configuration flag says a thing is possible; the flag's value in the authors' own largest
  recipe says whether it is worth it.
- **Price a wall-clock lever's cost in the currency it is actually paid in.** Async's 1.36x
  costs nothing in wall clock and everything in gradient correctness; a table with only
  seconds in it cannot show that.
- **Two levers acting on one quantity are not multiplicative, and their order changes the
  sum.** Ratios computed independently against the same baseline invite exactly that error.
- **A void number in a rigorous report is more dangerous than elsewhere**, because the code
  citations around it lend it credibility it no longer has. Deleting it and writing
  "unmeasured" is the repair; keeping the qualitative conclusion it supported is fine when
  that conclusion needs only its sign.
