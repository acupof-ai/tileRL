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

The roofline says the gap is larger than that. The per-tick stream, dumped per
key off the loaded model and closing to 0.002 GB against the resident total:

| what | GB |
|---|---:|
| `w8` (fp8 weights) | 10.625 |
| `wq` (fp4 nibbles) | 7.499 |
| `scale` (block scales) | 3.746 |
| `embed_tokens` | 2.543 |
| everything else | 0.026 |
| **total resident** | **24.439** |
| **streamed per decode tick** (embedding is one row, not the table) | **21.896** |

At 4.0 TB/s that is **5.47 ms/tick**. Long-tail idle explains roughly a factor of
two of the rollout gap.

**The rest is the model forward, and it runs at a fifth of bandwidth.** A first
breakdown on card 6 (2026-09-08, 27B, group 8, gen 1024) attributed 89.5% of the
rollout wall to `_sample_commit` and left the forward at 3.07 ms/tick. That
attribution was wrong, and the discriminator was the floor: **3.07 ms is half of
the 6.11 ms it takes to read the weights once.** A forward cannot finish in half
the time needed to stream the weights it multiplies. The decode path contains no
synchronisation between `_model.forward` and the `.tolist()` that reads the
sampled tokens back, so CUDA launches asynchronously, the forward timer stops at
the last kernel *enqueue*, and the whole forward's drain is billed to the first
host read — which is the sampler.

Direct microbenchmarks on the card settle it. Every operator inside sampling,
timed on H20 with a peaked logits fixture (nucleus 39 of 248320):

| operator | H20 | same on CPU |
|---|---:|---:|
| `sort` over V, descending | 0.274 ms | 20.53 ms |
| per-row Generator + `multinomial` x8 | 0.790 ms | 18.69 ms |
| `log_softmax(B,V)` | 0.055 ms | 0.52 ms |
| **all sampling operators** | **1.065 ms** | ~40 ms |

**Sampling is 3.6% of a 29.71 ms tick.** The other 28.64 ms is the forward, which
puts it at 5.23x the 5.47 ms floor — **19.1% of HBM bandwidth**. That is the
number this project should be optimising, and it is inside the decode kernel:
occupancy, KV traffic, GDN state. Not the sampler, and not a new process
topology.

**Two method results worth more than the number.** First, the CPU could not have
priced this: the logits are 8 x 248320 x 4 B = 7.95 MB, a radix sort moves about
0.16 GB, and at 4 TB/s that is ~40 us — so the CPU's 20.53 ms is **517x** the
GPU's bandwidth floor. An O(V) mechanism whose V is only a few megabytes is
compute-bound on CPU and bandwidth-bound on GPU; measuring it on CPU is wrong by
orders of magnitude, not by a constant. Second, the finding that survived was the
one whose author wrote down its own falsifier: "if `sort` over V is 1-2 ms on the
card, this is not the mechanism." It measured 0.274 ms.

The sampler does hold two real but small wins — batching the per-row
`multinomial` loop (9.35x on that operator, 2.6% of the rollout) and topk in
place of the full sort (0.6%). Together 3.2%, and they wait until
`steps_to_score` exists to price them.

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

3. **Attack the decode kernel's bandwidth utilisation.** It is 19.1% of the
   floor and it is 96% of the rollout tick. The sampler, which the first
   breakdown named, is 3.6%.

**A third method result, and the reason the number above is 19.1% rather than the
21.8% this document carried an hour earlier.** Two sessions derived the weight
bytes from shape independently and landed **1.9% apart** (18.23 and 18.58 GB)
against a measured 24.44 GB. The agreement read as confirmation. It was not: both
started from `fp4_param_keys` and both encoded "every quantised linear is fp4",
and the missing 5.86 GB was `w8` — the checkpoint is mixed precision and its fp8
weights carry more bytes than its fp4 ones. **Two derivations agreeing measure
their shared premise, not their conclusion.** A cross-validation has to differ in
kind — a measurement against a derivation — and the measurement here was one dump
of `numel * itemsize` per key.

**What this document cannot yet decide.** Why the decode forward sits at 19.1%
of bandwidth rather than near it. Occupancy, KV traffic and the GDN state are the
candidates and none has been measured. Until one is, "the forward is 4.69x the
floor" is a bound, not a diagnosis.

## Sources

- https://www.anyscale.com/blog/open-source-rl-libraries-for-llms
- https://arxiv.org/pdf/2405.11143 (OpenRLHF)
- https://arxiv.org/html/2509.21009 (RollPacker, long-tail rollouts)
- https://arxiv.org/html/2509.18521v1 (APRIL, active partial rollouts)
- https://arxiv.org/pdf/2606.26997 (RolloutPipe)
- https://arxiv.org/pdf/2604.26256 (DORA)
