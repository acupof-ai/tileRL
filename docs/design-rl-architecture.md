# What the RL training architecture should be — survey and choice

Written 2026-09-08 because the project had no such document. Four design docs
existed (engine, kernels, RL stack, parallel) and every one of them argues from
the layer below it. Nothing argued from the top: what wall clock does a given
score cost, and what architecture minimises it.

The objective, set by ckl on 2026-09-08:

```
time_to_score = steps_to_score  x  seconds_per_step
                (algorithm)        (infrastructure)
```

A denominator lever multiplies; a numerator lever adds. That ordering is the
only thing in this document that ranks work.

## Where we actually are

| quantity | value | source |
|---|---|---|
| `seconds_per_step` | 85.617 s | `wins/2026-09-07-the-step-is-74-percent-rollout.md`, 27B, group 8, gen 1024 |
| rollout share | 73.77% | same |
| `steps_to_score` | **never measured** | both 27B RL runs failed mechanically |

The product is therefore unbounded, and every infrastructure number in this repo
is currently multiplied by an unknown.

## What the field does

Rollout dominance is not our defect. Published RL post-training systems report
the rollout phase at **more than 70% of total training time**, ahead of backward
and optimizer combined. Our 73.77% is the normal operating point, not a
pathology — which means the levers other people found are the levers available
to us.

Two architectural families:

- **Colocated** — inference and training share the GPUs and run in sequence.
  verl, OpenRLHF, ROLL. Simple, no weight transfer, but the card is idle for
  whichever half is not running, and verl's HybridEngine has to reconfigure the
  tensor-parallel layout on every switch.
- **Disaggregated** — rollout runs on separate workers, weights are shipped.
  slime (SGLang + Megatron), AReaL, DORA. Costs a weight sync per step, buys
  overlap and lets the rollout side scale independently.

tileRL is colocated and goes further than any of them: **one process, one engine,
one copy of the weights, serving and training**. That removes the weight sync
entirely — a real advantage, and the reason `invalidate_secs` is 0.001 s where a
disaggregated system pays a full model transfer.

The cost of that choice is that we inherit colocated's idle time with no
mechanism to hide it, and we have never measured what that costs.

## The identified bottleneck, in the field and in our code

The field's diagnosis of the rollout phase is the **long tail of generation
lengths**: a group finishes when its slowest member finishes, so workers idle
behind stragglers, and utilisation falls sharply after the first hundred decode
steps. The published responses are partial/interruptible rollouts (AReaL),
tail-aware packing (RollPacker), active partial rollouts (APRIL) and per-request
load balancing (slime).

Our code has this in its purest form. `train.py:209`:

```python
prompt = np.asarray(prompts[step % len(prompts)], dtype=np.int64)
ids = [engine.submit(prompt.tolist(), ...) for g in range(group)]
# then: tick until every id is done
```

**One prompt per step**, `group` completions of it, and the step ends when the
last one ends. There is no other work in the queue, so every row that finishes
early leaves its slot idle for the rest of the step. The step's tick count is
the longest completion's length by construction.

Run 2's recorded means — 890, 1122, 1198, 1277, 1923 tokens across steps — put
the within-step spread in the range where this costs a factor near two.

The roofline says the gap is larger than that. 27B NVFP4 weights are 24.44 GB;
at 4.0 TB/s a decode tick that reads them once takes 6.11 ms, so a group of 8
has a ceiling of 1309 tok/s. Measured rollout is 8192 tokens in 63.156 s =
**130 tok/s, 0.099 of the ceiling**. Long-tail idle explains roughly a factor of
two of that.

**The rest is not the model.** A first breakdown on card 6 (2026-09-08, 27B,
group 8, gen 1024, two pooled steps) puts `_run_forward` at 99.9% of the rollout
wall and `_sample_commit` — nested inside it — at **89.5%**: 27.2 s of a 30.4 s
step. The model forward is the remaining ~3.1 s. Sampling, a host-side
logits-to-token path, is the largest single block in the step, and folded back
into the 85.617 s profile it is on the order of two thirds of the whole GRPO
step. Nothing in the roadmap, the design docs or six card-sessions of rollout
work had this as a candidate.

Two caveats hold this number below a verdict. The probe's conservation assertion
was vacuous — the remainder was *defined* as `wall - sum(parts)` and then
asserted to close, so a **negative** remainder of -27.2 s printed as data instead
of raising. And the probe's aggregate rate (270 tok/s) is 2x the profile's
(130 tok/s), unexplained, so its absolute seconds cannot yet be aligned with the
85.617 s table. The nesting does not threaten the headline — a child at 89.5% of
a parent that is 99.9% of the wall is a valid share either way — but the entry
that records it must come from the fixed instrument.

## The choice

**Keep one process and one weight copy.** It is our genuine differentiator, it
is why our weight-update cost is ~0, and nothing in the survey argues against it
at single-card scale. Disaggregation earns its keep when rollout and training
want different amounts of hardware; on one card they cannot.

**Take the field's rollout fix, not its process topology.** The two changes that
follow from the survey and cost nothing architecturally:

1. **More than one prompt per step.** This is the highest-value change in the
   document. It fills the long-tail idle slots with useful work, and it
   independently fixes the failure that killed run 1 — a group of 8 attempts at
   one question is solved 8/8 or missed 8/8, which is why 73 of 81 tied steps
   tied at the ceiling and no gradient existed. One change, two known failure
   modes.
2. **Do not wait for the whole group.** Score and advantage need the group, but
   the engine does not need to idle: a finished row's slot should take the next
   prompt's rollout rather than wait.

3. **Attack the sampler before any kernel.** On the first breakdown it is the
   largest block in the step by a wide margin, and it is host-side Python and
   tensor plumbing rather than a scheduled kernel — the cheapest class of thing
   to fix. This displaces the kernel and KV work that six card-sessions went to.

**What this document cannot yet decide.** What is *inside* the 27.2 s: a host
sync, a per-row `.cpu()`, a Python loop over the batch, or genuine compute. That
bucket needs its own decomposition, and until it has one, "sampling is 89.5%" is
a name standing in for a mechanism — the same shape as `backward_secs` being
19.7% forward, which is the error this document's own headline number exists to
correct.

## Sources

- https://www.anyscale.com/blog/open-source-rl-libraries-for-llms
- https://arxiv.org/pdf/2405.11143 (OpenRLHF)
- https://arxiv.org/html/2509.21009 (RollPacker, long-tail rollouts)
- https://arxiv.org/html/2509.18521v1 (APRIL, active partial rollouts)
- https://arxiv.org/pdf/2606.26997 (RolloutPipe)
- https://arxiv.org/pdf/2604.26256 (DORA)
