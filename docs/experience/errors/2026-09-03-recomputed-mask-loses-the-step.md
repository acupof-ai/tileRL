---
question: Why would recomputing the sampler's truncation mask at training time lose 96% of GRPO steps?
status: measured
source: tileRL, measured on the tiny model while scoping docs/rl-sota-parity.md §2
---

# A per-position error of well under 1% loses most of the steps

The plan was to fix the sampler/policy mismatch by recomputing `top_k`/`top_p`
at training time, so the gradient would be taken under the same truncated
distribution the rollout was drawn from. That is the right objective and the
wrong mechanism. Recomputing the mask instead of carrying it would have
discarded most training steps, silently.

## The mechanism

The rollout forward is prefill plus paged-KV decode. The training forward is
`_training_kv` dense full-width (`train.py:24-38`). They are not bitwise equal,
so a token near the nucleus boundary can fall inside the sampler's kept set and
outside the trainer's. When that happens the trainer computes `q(a) = 0` for the
token that was actually drawn, so `log q = -inf`, the loss is `inf`, and `_step`
returns early at `train.py:72-73` without updating anything.

## Measured, tiny model, card sampler (T=0.7, top_p=0.8, top_k=20)

1920 positions, 10 steps x 8 rollouts, fresh engine per step:

| | |
|---|---|
| `top_k=20` kept-set misses | **0 / 1920** (rank-order is robust) |
| `top_p=0.8` nucleus misses | **3 / 1920 = 0.156%** |
| score gap engine vs dense | max 0.271, mean 0.0097 nats |
| boundary token's q | mean 0.269, max 0.500 |

**The rate is not pinned.** Three independent probes measured the sampled-token
drop rate as 3/1920 (0.156%), 1/512 (0.20%), and 1/2160 (0.046%) — every one a
single-digit event count, spanning 4x. The conclusion survives across the whole
range, which is why it is stated as a range:

| completion length | scored positions (group=8) | P(step lost) at 0.046% – 0.20% |
|---|---|---|
| 16 | 128 | 5.7% – 22.6% |
| 32 | 256 | 11.1% – 40.1% |
| **256** | **2048** | **61.0% – 98.3%** |

`grpo-gsm8k-27b` is `group=8, max_new_tokens=256`. Even at the lowest measured
rate, most steps are lost.

The failure would present as "GRPO runs but never learns". `inf` prints as a
loss value; nothing raises.

## Two things this cost me

**The position rate is not the step rate.** 0.156% reads as negligible and it is
the wrong denominator. A step is lost if *any* of its scored positions drops its
token, so the rate compounds over `group x length` and reaches near-certainty at
the recipe's own settings. Any per-token error rate in a training path has to be
converted to a per-step rate before it can be called small.

**My planned test could not have caught it.** The gate I intended was a bitwise
no-op check at `top_k=0, top_p=1.0, T=1.0` — proving the masked gradient equals
the current one when truncation is off. It passes, and it is worthless here:
disabling truncation is exactly the configuration where a dropped token cannot
occur. A guard that constrains only the case where the mechanism is absent is
not a guard.

## The fix, which has a name

Carry the rollout's kept set into the gradient rather than recomputing it.
Converged on independently by three stacks in the month before this was written:

- **DeepSeek-V3.2 §3.1, "Keep Sampling Mask"** (arXiv:2512.02556): "we preserve
  the truncation masks during sampling ... and apply them to [the current
  policy] during training, ensuring both policies share identical action
  subspaces." Qualitative; no ablation.
- **vLLM PR #49577 "Mask Replay"**, merged 2026-08-13: exposes per-token kept
  support as CSR. Measured mean importance ratio ~1 with replay; "consistently
  below 1" with a "persistent negative bias" without.
- **prime-rl #3235 -> #3431**, merged, cites DeepSeek explicitly. Trainer math:
  `logprob = logits[label]/T - logsumexp(logits[kept]/T)` at masked positions,
  full-vocab `logsumexp` elsewhere. Measured at `top_p=0.97`: reward 0.19->0.75
  vs baseline 0.18->0.73, i.e. parity.

prime-rl's history is the useful part: they hardcoded truncation **off** for
train rollouts first, ran on that, and only then built mask transport. tileRL
took the same order for the same reason — the recipe has never completed a step,
so there is nothing for a subtler estimator to be measured against.

## Rule

**Convert a per-token rate to a per-step rate before calling it small.** The
unit that matters is whatever unit can be lost whole.

And: before crediting a guard, name the configuration in which the bug fires and
check that the guard runs there. A no-op test that switches off the feature under
test is not evidence about the feature.

## Unmeasured

Every nucleus-width number here is the tiny model (vocab 320) or synthetic at
V=248320. Nothing was read off the real checkpoint — `TILERL_QWEN38_SOURCE` is
unset on this machine and the weights live on the pod at
`/work/Qwen3.8-27B-NVFP4`. The mechanism does not depend on the frequency; the
frequency is `pending-remote`.
