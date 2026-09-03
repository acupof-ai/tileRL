# tileRL's RL loop against TRL and AReaL

Phase 1: read both stacks at source level, put every default beside ours, and
say for each difference whether we chose it or missed it.

Outside half read on 2026-09-03 from the checkouts, not the READMEs:
huggingface/trl `312727b3` (VERSION 1.13.0.dev0) and inclusionAI/AReaL
`24e2ea43`. tileRL half is `a702c9a`. Where an AReaL dataclass default and its
shipped example YAML disagree, both are given — in AReaL that gap is usually
the whole story.

The short version. We are **stricter than both on on-policy discipline** and
that is the structural bet the whole project rests on. Our sampler was truncated
(`top_k=20`, `top_p=0.8`, `T=0.7`) while the gradient was taken under the full
softmax, which biased every step run before 2026-09-03; **RL rollouts now draw
untruncated, so the sampler is the policy.** That is a waypoint, not the
destination — using the card's sampler in RL needs the rollout's kept set carried
into the gradient, which three stacks shipped in the month before this was
written and we have not. Our memory story is where both stacks did engineering
and we did none, and it is why `grpo-gsm8k-27b` cannot finish a step.

§2 has been amended twice since it was first written; both amendments are against
the original text. Its claim of having no precedent was wrong, and its claim that
we had the ingredients for importance weighting was half wrong — the ingredient
was defective.

---

## 1. Rollout/training weight sharing and on-policy discipline

| | TRL | AReaL | tileRL |
|---|---|---|---|
| rollout engine | vLLM, `vllm_mode="colocate"`, `use_vllm=False` (`grpo_config.py:583,587`) | SGLang, separate process | our own `Engine`, same process (`engine.py:253`) |
| weight transfer | `sync_weights` one param at a time, PEFT merge/unmerge inside a full gather (`vllm_generation.py:481`) | `weight_update_mode="xccl"`, 1024 MB buckets (`cli_args.py:1337`) | **none.** `build_engine` stores the caller's `Model` (`engine.py:263`) and reads `model.params` at every tick (`engine.py:630`); optimizers write those tensors in place (`autograd.py:427`) |
| staleness | one sync per optimizer step; `num_iterations=1`, so the stored old logprobs are `None` and the PPO ratio is exactly 1 | `max_head_offpolicyness=0` in the dataclass, `2` in `gsm8k_grpo.yaml:29`; admission control only — a finished-but-stale trajectory is never rejected | one update per rollout, always |
| prefix cache | flushed unconditionally at the end of every sync (`vllm_generation.py:512-517`) | no application-level flush; relies on SGLang internals | `NoPrefixStore` never matches and never retains (`kv_cache.py:289-302`); `_require_on_policy` raises if you pass anything else (`train.py:171-176`) |
| decode CUDA graphs | `enforce_eager` never passed in colocate — graphs stay captured across weight updates, nothing recaptures them | not addressed | `decode_graph=False` required; the same guard raises |

**Verdict: deliberate, and stricter than either.** TRL solves the cache problem
by flushing and leaves the graph problem alone; AReaL does not treat staleness
as a correctness problem at all. We removed the problem instead of correcting
it: there is no second copy of the weights to go stale.

The price is on the roadmap, not hidden: `# ponytail: recapture the graph and
drop the prefix entries after each update instead of disabling both`
(`train.py:199-200`), roadmap P2.0. Today an RL rollout decodes without a
captured graph, which the spec-decode work measured at 86.2 tok/s captured vs
17.6 uncaptured.

## 2. The behaviour policy is not the policy we differentiate

This is the sharpest gap and it is ours alone.

`grpo_loop` samples with whatever `SamplingParams` the caller passes. For the
27B recipe that is `prompt.sampling(...)`, which loads the model card's
values — thinking off: `temperature=0.7, top_p=0.8, top_k=20`
(`prompt.py:14-15`). `rl_step` then takes the causal cross-entropy gradient
under the **full, untempered softmax** (`train.py:154`, `reference.py:812`).
The samples come from π truncated to 20 tokens and renormalized at T=0.7; the
score function is ∇log π. Nothing reweights one to the other.

| | correction |
|---|---|
| TRL | samples at `temperature=1.0, top_p=1.0, top_k=0` — the sampler *is* the policy. Separately, `vllm_importance_sampling_correction=True` by default with `clip_max=3.0` (`grpo_trainer.py:2678-2723`) corrects the much smaller vLLM-vs-train numeric gap |
| AReaL | decoupled PPO: `ratio = exp(logprobs - proximal_logprobs)` (`functional.py:452`); the behaviour-policy weight is applied only when `rejection_sampling` is configured, and `cli_args.py:1943` warns when it is not |
| tileRL | none |

We had the ingredients and they were broken. `SamplingParams.logprobs` produces
per-token log-probs (`engine.py:856-864`) and `Engine.logprobs()` returns them
(`engine.py:420-432`), but `_restrict` applied `top_k` and not `top_p` while the
sampler applied both, so the reported score was low by exactly
`-log(kept top_p mass)` — +0.212 nats mean at this vocab, and a value of −0.054
for draws whose true log q was 0.0. Fixed 2026-09-03
(`errors/2026-09-03-logprobs-skipped-top-p.md`). `grpo_loop` still never sets the
flag; it does not need to under the untruncated sampler, and it will when mask
transport lands.

**Verdict: omission. Fixed as of 2026-09-03 by sampling untruncated during RL
rollouts** (`train.untruncated`, forced in `grpo_loop`), which makes the sampler
the policy by construction. Serving keeps the card's values.

**`opd_loop` needs no equivalent, and this is not an oversight.** It takes plain
cross-entropy on the teacher's sampled sequence through `train_step` — maximum
likelihood on given tokens, not a score-function estimator. The teacher's
truncation chooses *which* completions to distill, a data-quality decision; there
is no measure to mismatch. `untruncated()` belongs in `grpo_loop` alone.

**How far the bias went, measured.** Not reversed — attenuated and rotated. Over
6 configurations (3 model seeds x 2 prompt sets), 64 advantage draws per position:
mean projection of the shipped gradient onto the true policy gradient **0.72–0.90**
(so roughly 20% of the magnitude lost), cosine down to 0.20 at the worst
positions, and **~11% of individual advantage draws** give a negative projection.
Positions where the projection is negative for a *majority* of draws: 0/12 in every
configuration. An earlier single-draw probe reported 2/12 and that was its own
variance.

**Three ways out, not two, and the third is where the field went.** This section
originally read as if tileRL were alone here. It is not — the row was written
before three stacks converged on the same answer, all in the month before:

| stack | what they do | measured? |
|---|---|---|
| **DeepSeek-V3.2 §3.1** "Keep Sampling Mask" (arXiv:2512.02556) | "preserve the truncation masks during sampling ... and apply them to [the current policy] during training, ensuring both policies share identical action subspaces" | no — qualitative, no ablation, rollout values undisclosed |
| **vLLM PR #49577** "Mask Replay", merged 2026-08-13 | exposes the per-token kept support as CSR so the trainer renormalizes over the same set | yes, distribution alignment only: ratio ~1 with replay; "consistently below 1" with a "persistent negative bias" without |
| **prime-rl #3235 -> #3431**, merged | same, citing DeepSeek. `logprob = logits[label]/T - logsumexp(logits[kept]/T)` at masked positions | yes: at `top_p=0.97`, reward 0.19->0.75 vs baseline 0.18->0.73 — parity |

vLLM's unreplayed measurement is worth noting: a ratio "consistently below 1"
with a persistent negative bias is the same sign as the `top_p` scoring defect
found here on 2026-09-03 (`errors/2026-09-03-logprobs-skipped-top-p.md`), from an
independent codebase.

**prime-rl shipped untruncated rollouts first, ran on them, then built mask
transport.** tileRL took that order for the same reason: with no completed GRPO
run, a subtler estimator has nothing to be measured against.

**The importance-weight route (AReaL's) is closed, with numbers.** Not merely
noisy — structurally unusable for a truncated sampler. Measured at V=248320,
`T=0.7, top_p=0.8, top_k=20`:

| | |
|---|---|
| π mass outside q's support, per position | **97.2%** — ratio is infinite there, unreachable by any weight |
| ESS over a group of 8, L=8 / 32 / 256 | 5.50 / 3.51 / **1.74** |
| mean sequence log-ratio at L=256 | −921.6 |
| fires at TRL's `clip_max=3.0` / AReaL's `upper=5.0` | **0.0% / 0.0%** |

The clip never fires because every weight sits far *below* 1. Both stacks' caps
are upper bounds on a weight that collapses downward, so the guardrail is on the
wrong side. A group of 8 becomes under 2 effective samples at the recipe's own
length. This is why the ratio+clip marker at `train.py:146` is about rollout
reuse only — it was never going to carry this correction.

**Recomputing the mask at train time is not the same as transporting it**, and
the difference is 96% of the steps:
`errors/2026-09-03-recomputed-mask-loses-the-step.md`.

## 3. The loss

| | TRL | AReaL | tileRL |
|---|---|---|---|
| objective | `loss_type="dapo"`: `-min(r·A, clip(r,1-ε,1+ε)·A)`, `epsilon=0.2`, `importance_sampling_level="token"` | `max(-A·r, -A·clip(r))`, `eps_clip=0.2`, optional `c_clip`, optional rejection-sampling mask (`level="token"`, `upper=5.0`) | `-A · ∇log π`, no ratio, no clip (`train.py:152-168`) |
| rollout reuse | `num_iterations=1`, so the ratio is inert out of the box | `ppo_n_minibatches=4` — every rollout is reused four times | one update per rollout |
| normalization | global active-token count over the whole generation batch, rescaled to one accumulation window | global per-token sum, denominator captured *before* rejection sampling narrows the mask, then all-reduced over the DP group | scored-token count over the whole batch, `n` at `train.py:164` |
| KL | in the loss, k3 estimator, `beta=0.0` default → **the reference model is never loaded** | in the reward, k1, `kl_ctl=0.1` default but **0.0 in every math example** | none anywhere (`grep -i 'kl\|ref_model\|ref_logp' src/` is empty) |

**Normalization: aligned, deliberately.** Both stacks converged on the same
denominator and so did we. Ours differs only in what "the whole batch" means:
one prompt's group of 8, not a multi-prompt generation batch.

**KL: aligned with both stacks' effective defaults, not an omission.** TRL
ships `beta=0.0` and AReaL ships `kl_ctl: 0.0` in every math recipe. Nobody is
running a KL penalty on verifiable-reward tasks. Ours being absent costs
nothing today; it becomes a gap the day a run needs to be held near the base
model.

**Ratio and clip: omission, already marked.** `train.py:146-147` carries
`# ponytail: single-update REINFORCE-with-baseline; add the PPO ratio+clip when
a rollout is reused for more than one step.` That marker states the coupling
correctly: no ratio ⟹ μ must be 1. What it does not say is that §2 makes the
ratio load-bearing *even at μ=1*, because our sampler already differs from the
policy. The clip is optional; the importance weight is not.

## 4. Advantage estimation

| | TRL (`grpo_trainer.py:2777-2805`) | AReaL | tileRL (`train.py:122-128`) |
|---|---|---|---|
| baseline | group mean | group mean via `reward_norm` (YAML, `None` in the dataclass) | group mean |
| scaling | `scale_rewards="group"` → divide by group std | `gsm8k_grpo.yaml`: mean group, std group | divide by group std |
| std == 0 | no special case; numerator is exactly 0 so the advantage is 0. Logged as `frac_reward_zero_std` | `(x-mean)/(std+1e-5)` → 0 | `np.where(std > 1e-8, std, 1.0)` then `np.where(std > 1e-8, adv, 0.0)` — explicitly zero |
| degenerate groups | **not filtered.** No DAPO dynamic sampling anywhere in `trl/trainer/` | **not filtered.** `drop_incomplete_group=False` drops on rollout failure, not on constant reward | not filtered, but **counted and gated**: `tied_group_fraction` is a run metric and `groups_untied` fails the run above 0.5 (`cli.py:254`) |
| second whitening | none | every example adds `adv_norm: {mean batch, std batch}` on top | none |

**Verdict: aligned, and one place we are ahead — with a caveat that may reverse
it.** Nobody filters degenerate groups; we are the only one of the three that
makes "the group tied" a gate rather than a log line. The missing second
batch-wide whitening is a real difference — with one prompt per step there is no
batch to whiten over.

The caveat, measured 2026-09-03: **an all-or-nothing reward ties every group, at
every completion length** (36/36 group-steps on the tiny model, vs 0/12 under a
graded reward). GSM8K is all-or-nothing, so the gate is expected to fire on the
first steps of a cold LoRA — for a reason that is not a defect, with an exit code
indistinguishable from a wrong gradient. A gate that fires on the expected first
step is a gate that gets switched off by whoever hits it. TRL's choice to log
`frac_reward_zero_std` instead may be the better one for exactly the run we are
about to make. Open question for ckl, not a proposal:
`errors/2026-09-03-tied-groups-are-the-rewards-shape.md` carries the numbers and
the discriminator.

## 5. Memory — the measured failure

Both stacks name the vocab-sized logits tensor as the peak and both only
mitigate it. Our vocab is 248320, larger than either.

| | TRL | AReaL | tileRL |
|---|---|---|---|
| gradient checkpointing | `gradient_checkpointing=True` (TRL overrides HF's `False`) | `False` in the dataclass, `true` in every example | **none** |
| micro-batching | `gradient_accumulation_steps`; `auto_find_batch_size` **rejected outright** because halving the batch breaks prompt-group integrity | `max_tokens_per_mb=None` → 1e12, i.e. no split out of the box; every example sets 10240 | **none** — the whole group goes through one backward |
| logits handling | `_get_per_token_logps_and_entropies` chunks rows, still materializes `[b, C, V]`; `selective_log_softmax` loops row by row; the real fix is Liger's fused chunked LM head, which never materializes `[B,T,V]` | `logprobs_chunk_size=1024`; vocab-parallel custom autograd whose backward **overwrites the saved softmax in place** as `onehot - softmax`, allocating no new large tensor | `cross_entropy_loss_grad` holds **four** `[B,T,V]` f32 tensors at once (`reference.py:817-824`), then `rl_step` allocates a fifth for `grad * w * scale` |
| optimizer state | HF defaults, 8-bit/paged available | `optimizer_dtype="float32"` | fp32 moments (`precision.py:12`), but only over LoRA adapters |

**Verdict: omission, and it is the reason the recipe dies.**
`tilerl train --recipe grpo-gsm8k-27b` (group 8, 256 new tokens, LoRA rank 16)
reaches the training step and raises `torch.OutOfMemoryError` allocating a
146 MiB buffer with 95.15 of 95.22 GiB already used. Weights are ~23 GiB, so
the rest is activations the tape is holding.

TRL's reason for rejecting `auto_find_batch_size` is the reason the fix cannot
be "use a smaller group": the group is the baseline. Micro-batching with
gradient accumulation is the only shrink that leaves the advantage
normalization untouched. Fixed on `train/grpo-memory`, with the equivalence
proved as a test rather than argued.

## 6. Sequence handling

| | TRL | AReaL | tileRL |
|---|---|---|---|
| layout | prompts left-padded, completions right-padded; no packing path for GRPO | `pack_tensor_dict` → `[total_length, ...]` with `cu_seqlens`, padded to 256-token pages, FA-2 varlen | right-padded to `max(len(completion))` (`train.py:228-235`); no packing anywhere (`grep cu_seqlens src/` is empty) |
| loss mask | prompt 0, completion 1 | same, plus prompt `version=-1` | scored iff `prompt_len <= i+1 < seq_len` (`train.py:157-163`) |
| truncated completions | `mask_truncated_completions=False` — trained on; rate logged as `completions/clipped_ratio` | `mask_no_eos_with_zero=False` — reward zeroed, tokens still trained | trained on, **rate not measured** |
| over-long prompts | `max_prompt_length` removed from `GRPOConfig` entirely — no truncation, pre-filter your dataset | filtered out of the dataset, not truncated | neither truncated nor filtered |

**Padding vs packing: deliberate for now.** One prompt per step and a group of
8 completions of similar length means padding waste is small; `seq_q_lens`
exists on the engine path and `_training_kv` never sets it (`train.py:24-38`),
so the training forward always sees full width. Packing becomes worth it when
a step covers several prompts.

**The clipped-completion rate is an omission.** Both stacks measure it because
a rising fraction of length-truncated rollouts is how a GRPO run degenerates
without the reward moving. We have `max_new_tokens=256` and no counter.

## 7. Optimizer and schedule

| | TRL | AReaL | tileRL |
|---|---|---|---|
| optimizer | HF `adamw_torch_fused` | `adam`, `weight_decay=0.01`; `adam_bf16` variant with bf16 moments and Kahan summation | `AdamW(betas=(0.9,0.95), eps=1e-8, weight_decay=0.1)` (`cli.py:193`); `Adafactor` and `ISO` are full-parameter SFT only |
| learning rate | **`1e-6`** — GRPO's only override of HF's `5e-5` | dataclass `1e-3`, **examples `6e-6`–`1.7e-5`** | **`1e-3`** (`cli.py:453`) |
| schedule | linear decay | `constant`, `warmup_steps_proportion=0.001` | **none.** `cosine_warmup` exists (`autograd.py:512`) and the RL path never calls it |
| grad clip | `max_grad_norm=1.0` | `gradient_clipping=1.0`; a non-finite grad norm drops the whole step | `clip_grad_norm(grads, 1.0)`, hardcoded (`train.py:103`); non-finite norm skips the update |
| reference under LoRA | `beta==0` or `is_peft_model` → `ref_model=None`; adapter toggling stands in | a separate full model; `disable_adapter` appears nowhere | no reference at all |

**The learning rate is the row to argue about.** We train rank-16 LoRA
adapters, which legitimately carry a higher rate than full-parameter RL, so
`1e-3` is not the 1000× error it looks like next to TRL's `1e-6`. But it is
still roughly an order of magnitude above the usual LoRA RL range, it is
paired with `weight_decay=0.1`, and **nothing in this repo has measured it** —
the 27B recipe has never completed a run. Flag, not a verdict.

**No schedule: omission**, and cheap — `cosine_warmup` is already written.

## What to take from this

| | |
|---|---|
| stricter than both | on-policy discipline: no weight sync, no prefix cache, no captured graph (§1) |
| ahead of both, pending a caveat | a tied-group gate rather than a log line — but a sparse reward ties every group, so it fires benignly on step 1 (§4) |
| aligned on purpose | loss normalization, group-std advantages, no KL, no prompt truncation |
| **fixed 2026-09-03** | the sampler is now the policy: RL rollouts draw untruncated (§2). Waypoint — the destination is mask transport |
| behind the field, next | mask transport, so the card's sampler can be used in RL. Three stacks shipped it in the month before (§2) |
| closed with numbers | importance weighting: 97.2% of π's mass outside q's support, ESS 1.74/8 at L=256, clips fire at 0.0% (§2) |
| omission, blocks runs today | no gradient checkpointing, no micro-batching, five vocab-sized tensors per step (§5) |
| omission, cheap | no LR schedule; no clipped-completion metric |
| deferred with a marker | PPO ratio and clip for rollout reuse (`train.py:146`); graph recapture after update (`train.py:199`) |

Every number in §2 is the tiny model (vocab 320) or synthetic at V=248320.
Nothing here was read off the real checkpoint: `pending-remote`.
