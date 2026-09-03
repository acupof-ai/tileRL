# tileRL's RL loop against TRL and AReaL

Phase 1: read both stacks at source level, put every default beside ours, and
say for each difference whether we chose it or missed it.

Outside half read on 2026-09-03 from the checkouts, not the READMEs:
huggingface/trl `312727b3` (VERSION 1.13.0.dev0) and inclusionAI/AReaL
`24e2ea43`. tileRL half is `a702c9a`. Where an AReaL dataclass default and its
shipped example YAML disagree, both are given — in AReaL that gap is usually
the whole story.

The short version. We are **stricter than both on on-policy discipline** and
that is the structural bet the whole project rests on. We are **missing the
entire off-policy correction layer**, and because our sampler is truncated
(`top_k=20`, `top_p=0.8`, `T=0.7`) while the gradient is taken under the full
softmax, that is not a dormant gap — it biases every step we have ever run.
Our memory story is where both stacks did engineering and we did none, and it
is why `grpo-gsm8k-27b` cannot finish a step.

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

We have the ingredients and do not use them. `SamplingParams.logprobs`
produces per-token log-probs of the sampled tokens (`engine.py:856-864`) and
`Engine.logprobs()` returns them (`engine.py:420-432`), but `grpo_loop` never
sets the flag and never reads them — a fact `wins/2026-08-29-grpo.md:16-17`
already recorded about an earlier version of the same field.

**Verdict: omission.** Two ways out, and they are not equivalent. Sampling at
`temperature=1.0, top_p=1.0, top_k=0` makes the sampler the policy and costs
nothing but rollout quality — TRL's choice. Keeping the card's sampler means
carrying an importance weight, which needs the per-token log-probs we already
compute — AReaL's choice, and the same machinery a PPO ratio would need.

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

**Verdict: aligned, and one place we are ahead.** Nobody filters degenerate
groups; we are the only one of the three that makes "the group tied" a gate
rather than a log line. The missing second batch-wide whitening is a real
difference — with one prompt per step there is no batch to whiten over.

## 5. Memory — the measured failure

Both stacks name the vocab-sized logits tensor as the peak and both only
mitigate it. Our vocab is 248320, larger than either.

| | TRL | AReaL | tileRL |
|---|---|---|---|
| gradient checkpointing | `gradient_checkpointing=True` (TRL overrides HF's `False`) | `False` in the dataclass, `true` in every example | `autograd.checkpoint`, on by default; the MLP block only — attention and GDN advance their pools and cannot be replayed |
| micro-batching | `gradient_accumulation_steps`; `auto_find_batch_size` **rejected outright** because halving the batch breaks prompt-group integrity | `max_tokens_per_mb=None` → 1e12, i.e. no split out of the box; every example sets 10240 | `rl_step(micro=N)`, `--micro`; the scored-token normalizer is the whole batch's, so the split is gradient-identical (`test_micro_batching_is_the_same_update`) |
| logits handling | `_get_per_token_logps_and_entropies` chunks rows, still materializes `[b, C, V]`; `selective_log_softmax` loops row by row; the real fix is Liger's fused chunked LM head, which never materializes `[B,T,V]` | `logprobs_chunk_size=1024`; vocab-parallel custom autograd whose backward **overwrites the saved softmax in place** as `onehot - softmax`, allocating no new large tensor | `cross_entropy_loss_grad` writes `softmax - onehot` over the logits and `rl_step` scales that buffer in place — one `[B,T,V]` f32 tensor, where the shape-for-shape version held five |
| optimizer state | HF defaults, 8-bit/paged available | `optimizer_dtype="float32"` | fp32 moments (`precision.py:12`), but only over LoRA adapters |

**Verdict: was the reason the recipe died; now aligned.**
`tilerl train --recipe grpo-gsm8k-27b` (group 8, 256 new tokens, LoRA rank 16)
used to reach the training step and raise `torch.OutOfMemoryError` allocating a
146 MiB buffer with 95.15 of 95.22 GiB already used. Weights are ~23 GiB, so
the rest was activations the tape was holding.

TRL's reason for rejecting `auto_find_batch_size` is the reason the fix could
not be "use a smaller group": the group is the baseline. Micro-batching with
gradient accumulation is the only shrink that leaves the advantage
normalization untouched, and its equivalence is a test rather than an argument
([wins/2026-09-03-grpo-27b-fits-the-card.md](experience/wins/2026-09-03-grpo-27b-fits-the-card.md)).
One row still reads differently than both surveys: the in-place CE is worth
0.25 GiB here, because our peak is the stored layer activations, not the vocab
tensor.

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
| ahead of both | a tied-group gate rather than a log line (§4) |
| aligned on purpose | loss normalization, group-std advantages, no KL, no prompt truncation |
| omission, biases runs today | the sampler is not the policy and nothing corrects for it (§2) |
| omission, blocks runs today | no gradient checkpointing, no micro-batching, five vocab-sized tensors per step (§5) |
| omission, cheap | no LR schedule; no clipped-completion metric |
| deferred with a marker | PPO ratio and clip (`train.py:146`); graph recapture after update (`train.py:199`) |
