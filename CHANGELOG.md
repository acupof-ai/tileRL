# Changelog

Central progress record. Three event classes land a line the same day, linking
the `docs/experience/` entry: **phase exit · default flip · accept-or-reject
verdict**. Newest first.

## 2026-09-02 — accept: ISO-Merger lands as `tilerl merge`, gated on two tiny specialists

- `src/tilerl/merge.py`: checkpoint-only merge of specialists sharing a base —
  frame displacements in the Stiefel tangent space, base spectrum kept, one
  ridge Gram solve for the coefficients (arXiv 2607.19331).
- Two tiny SFT specialists (batch A, batch B): loss A/B base **22.34 / 21.99**,
  average merge **18.46 / 17.55**, ISO **16.00 / 14.73**. K=1 returns the
  specialist to 3.3e-3; spectrum kept to 8.9e-7.
- 27B vs TIES/DARE: pending-remote.
  [wins/2026-09-02-iso-merger.md](docs/experience/wins/2026-09-02-iso-merger.md)
## 2026-09-02 — phase exit: the ISO optimizer runs on the tape (tiny); 27B pending-remote

- `tilerl.iso.ISO` wraps `Adafactor` / `AdamW`: every 2D weight trains as
  `U S V^T` with `S` frozen, frames retracted by Newton-Schulz after each
  step. Same `streams` / `begin` / `step_one` contract, so `train._step` is
  untouched. `tilerl train --optim iso` on the full-parameter SFT path.
- Gates: frame gradient vs finite difference **1.4e-10**, orthonormality
  **4.2e-7**, spectrum drift **5.9e-6** over 5 steps; loss curve matches
  Adafactor on the tiny model at 11 vs 7 ms/step.
- Not in `--rl` / `--opd` (LoRA, no ISO variant; full-parameter RL needs
  per-step fp4 re-quantization that does not exist). 27B memory and step time
  `pending-remote`.
  [wins/2026-09-02-iso-optimizer.md](docs/experience/wins/2026-09-02-iso-optimizer.md)
## 2026-09-02 — plan: the RL stack — ISO optimizer + merger, an on-policy DFlash2 head, a ledger CLI

- `docs/design-rl-stack.md`. ISO (arXiv 2607.19331) lands optimizer-side on
  the existing tape: frame gradients from the linear's `dW`, Newton-Schulz
  polar, Adafactor (AdamW on the frames is ~140 GB and does not fit a card),
  per-step re-quantization MMLU-gated. Its 2.7× fewer steps is 2.7× less
  rollout. ISO-Merger is `tilerl merge`, gated against TIES/DARE.
- The draft head: capture the speculative tick first (a draft disables graph
  capture, 86.2 vs 17.6 tok/s), then vendor DFlash2, then co-train it on the
  RL rollouts so it does not go stale. Speculation and batch are substitutes;
  one lever per regime.
- The ledger CLI is the agent's surface; the static page is last. Roadmap
  phases reordered P1–P6 accordingly; the P5 verdict rule now prices seconds
  to a target reward.

## 2026-09-02 — default flip: the north star is the serve+RL runtime on Hopper; serve throughput and ROCm are parked

- README, roadmap and AGENTS rewritten around what is measured and unoccupied:
  native NVFP4+FP8 decode on H20, where sglang/vLLM fall back to Marlin W4A16
  and recommend FP8, and an RL loop that samples from the served weights with
  no weight sync. Cross-platform is a test-harness property (every kernel has
  a CPU twin), not a product claim: ROCm's CPU-cell alias, its test and every
  mention are removed until a HIP host runs the suite. B≥8 / prefill work vs
  sglang is parked; rollout decode at B≥32 is the kernel that matters.
- `tilerl train --rl/--opd` takes real prompts (`--data` JSONL through the
  checkpoint tokenizer as ChatML), an exact-match GSM8K reward, `--eval-mmlu N`
  before and after on the same engine, and reports seconds per step.
  `scripts/rl_compare.sh` is the same-pod harness vs verl+sglang. The 27B
  numbers are **pending-remote**: the pod's eight cards hold another job.
  Gates: `test_train_cli_real_task`, `test_last_number`.
  [wins/2026-09-02-rl-real-task.md](docs/experience/wins/2026-09-02-rl-real-task.md)

## 2026-08-30 — accept: the sampler stops shipping its arguments to the device to read them back

- `temperature` / `top_p` / `seed` are Python scalars on `SamplingParams`. The
  engine put them on the device and `sample_batch` read them back to split the
  greedy rows from the sampled ones — 2 syncs a tick plus one per sampled row,
  on **every** target.
- Found by counting `aten._local_scalar_dense` in one decode tick on CPU, which
  takes the same fallbacks a partially-ported GPU target takes.
- Host syncs a decode tick: **7 -> 4** greedy, **15 -> 4** at B=8 with
  temperature > 0; of those, the ones that run on a target with the fused
  kernels go **3 -> 0**. Tick time is `pending-remote`.
- Draws unchanged (`test_sample_batch_matches_per_row`). Gate:
  `test_decode_tick_does_not_sync_in_the_sampler`.
  [wins/2026-08-30-sampler-host-syncs.md](docs/experience/wins/2026-08-30-sampler-host-syncs.md)

## 2026-08-30 — accept: the KV scatter fallback stops syncing once per token

- `PagedKvPool.write_tokens` looped per token with an
  `int(kv.block_table[...])` inside, and is called once per full-attn layer:
  `b * seq_q * layers` host syncs. **A 512-token prefill chunk of the 27B was
  8192 of them a tick.** `r.blocks` is a Python list the engine already holds.
- Indexed instead of looped. Prefill syncs 35 -> 1 on a 32-token chunk;
  decode 4 -> 1. The remaining one is the eager `gdn_forward` fallback.
- Counted with `aten._local_scalar_dense`, which does not see the two
  batch-wide `tolist()` that replaced the per-row scalars — those still
  transfer. Tick time is `pending-remote`.
- This is the fallback path for arches without the `write_tokens` kernel: the
  CPU target, and any partially-registered cell.
  [wins/2026-08-30-kv-scatter-synced-once-per-token.md](docs/experience/wins/2026-08-30-kv-scatter-synced-once-per-token.md)

## 2026-08-30 — phase exit: the draft head now attends over the whole prefix

- `_draft_chains` (a forward before each tick, against a one-block chain-local
  KV) became `_draft_step` (a forward at the END of each tick, over the
  request's own blocks). The draft runs at `[draft_pos+1 .. seq_len-1]` and its
  last position is the draft for the next token, so the KV fill and the draft
  are one forward and there is never a gap for its attention to read.
- Engine draft vs the probe's full-context draft, tiny model: **argmax matches
  everywhere**, residual norm-relative 1.3e-02 to 5.4e-02 — explained by the
  trunk hidden (paged/recurrent vs a dense re-derivation, 3.2e-03 to 4.1e-03,
  amplified ~10x by the head). Before: unrelated vectors, argmax 46 vs 232.
- Gate: `test_engine_draft_matches_full_context_draft` over four shapes
  (single / multirow / chunked / depth2), mutation-checked — forcing the fill's
  `seq_len` back to its own run length turns all four red.
- **Acceptance rate is `pending-remote`.** Whether the loop now reaches the
  probe's 84.4% and clears the 66% break-even needs a real model on a card.
  [errors/2026-08-30-draft-attention-sees-one-token.md](docs/experience/errors/2026-08-30-draft-attention-sees-one-token.md)

## 2026-08-30 — audit: verify_lens is dead on the shipped path and mis-modelled

- `trim=not self._decode_graph_on` and graph capture is on by default, so
  `_draft_chains` returns before reaching `verify_lens`. The arm that could say
  "do not speculate" is the one the shipped path skips.
- Its constants are agent-infer's (`BIAS_MS=211.0`, `ROW_MS=0.53`, marked
  "re-measure per target"). Measured here, the marginal cost of a verify row
  runs from +9.52 ms to **-0.11 ms** — mma8 pads M to 8, so rows 5-8 are free.
  No constant makes a linear model fit; the optimum is "fill the tile", not a
  survival threshold.
- Recorded only. Retuning needs a GPU run, and mixing readings across
  configurations is how this feature got a 1.9x optimistic estimate once.
  [errors/2026-08-30-verify-lens-is-dead-and-its-cost-model-is-wrong.md](docs/experience/errors/2026-08-30-verify-lens-is-dead-and-its-cost-model-is-wrong.md)

## 2026-08-30 — reopened: the draft head's attention sees one token in the loop

- `draft_check.py` measured 84.4% teacher-forced agreement by running the head
  over the whole sequence; `Engine._draft_chains` runs it with `seq_len = 1`, so
  its attention is a softmax over a single position and contributes nothing.
  The two acceptance numbers (84.4% and 55.8%) were never the same experiment.
- `_draft_kv` is one block per row, never filled with the prompt or with
  accepted tokens — there is no path by which the draft could see context.
- **This reopens the speculation verdict**: break-even is `p >= 66%`, the loop
  measured 55.8%, the probe measured 84.4%. Not a claim that fixing it wins —
  the fill pass has a cost and only a GPU run settles either number.
- Found by reading; nothing changed yet. Design outlined in the entry.
  [errors/2026-08-30-draft-attention-sees-one-token.md](docs/experience/errors/2026-08-30-draft-attention-sees-one-token.md)

## 2026-08-30 — phase exit: the spec suite's request budget bound mid-measurement

- `suite_spec` gave each request `(ticks + 20) * (1 + depth)` tokens against the
  ~70 ticks it actually runs. At B=1 depth 1 that is 80 against ~109 produced:
  the request finished around tick 51 of 70 and the empty ticks still counted,
  reporting **1.12 tok/tick where `1 + p` is 1.56**. Only the speculative arm
  hit it.
- Closes the open item in the speculation verdict, and explains its
  depth-1-only shape: depth 2's budget happened to clear.
- Corrected, B=1 depth 1 is 0.79x (was 0.57x) and B=8 depth 1 is 0.22x (was
  0.15x). **Speculation still loses everywhere and the break-even is
  unchanged** — it came from ms/tick and acceptance, which were sound.
- Budget now derives from `benchkit.SETTLE_BUDGET(b)`, and a row whose running
  set shrank during the window is voided rather than reported.
  [errors/2026-08-30-spec-suite-budget-bound-mid-window.md](docs/experience/errors/2026-08-30-spec-suite-budget-bound-mid-window.md)

## 2026-08-30 — accept: Adafactor stops syncing to the host once per parameter

- `Adafactor.step_one` called `float(tensor)` twice per parameter (update RMS,
  parameter RMS). 27B = 851 parameter tensors = **1702 host syncs a step**, each
  draining the pipeline mid-backward because `streams=True` interleaves the
  optimizer with the backward pass.
- Both RMS values now stay on device; the non-finite-gradient early return
  becomes a zero scale. Parameters match the old formula to 1.2e-07 over 5 steps.
- Host syncs a step: **55 -> 1** on the tiny model (the loss finite-check).
  Step time on the 27B is `pending-remote` — this host has no GPU and the win is
  a sync/overlap effect a CPU run cannot show.
- Gate: `tests/test_e2e.py::test_train_step_does_not_sync_per_parameter`.
  [wins/2026-08-30-adafactor-host-sync-per-parameter.md](docs/experience/wins/2026-08-30-adafactor-host-sync-per-parameter.md)

## 2026-08-30 — verdict: TP works, and a capturable all-reduce is its entry ticket

- First successful TP=4 run on the 27B, eager both sides: **1.44x at B=1**
  (10.9 -> 15.7 tok/s) and **1.60x at B=8** (57.9 -> 92.6). Not 4x, because
  128 all-reduces per tick cost ~2.8 ms at a measured 21.5 us small-message
  floor that does not amortise over batch.
- Against what SHIPS, on the same four cards, DP=4 with graph capture is
  373.6 / 1323.6 tok/s — **TP is 14x behind**, all of it the forfeited graph.
  `torch.distributed` collectives cannot be captured (SGLang's capture table
  lists it as the one backend that cannot), so `Backend.all_reduce` needs a
  pynccl-style path before TP can win anything. Not an optimisation: a
  prerequisite.
- Capacity is still why TP=4 x DP=2 beats TP=8 on this model — 4 KV heads
  means TP=8 stores the cache twice, and at depth 8192 concurrent requests
  are DP=8 826, TP=8 605, TP=4 x DP=2 1042.

## 2026-08-30 — phase exit: lm_head stopped running over every position of every row

- `last_only` was one bool, so it could only say "every row ends at the same
  position". A mixed tick's decode rows end at 1 while the prefill row spans
  the width, so it fell to False and lm_head — [248320, 5120], the largest
  projection in the model — ran over all rows at full width.
- Live tensors at B=32 opened with three `(3072, 248320)` f32 logits at
  **8.53 GiB** together, for six rows of which five needed one position each.
  Per-row lengths now. B=64 peaks at **39 GiB** where it used to OOM on 95.
  [wins/2026-08-30-lm-head-full-width-on-mixed-ticks.md](docs/experience/wins/2026-08-30-lm-head-full-width-on-mixed-ticks.md)
- Two more memory fixes behind it, neither yet measured on GPU: decode graphs
  capture on a size ladder instead of per exact batch (a draining batch
  captured up to 32 graphs), and all buckets now share ONE graph memory pool.

## 2026-08-30 — phase exit: offline batch generation, one process per device

- `tilerl generate` fans a prompt corpus across devices, a process each. Not
  `DataParallelEngine`: that runs N CUDA contexts in one interpreter and
  serialises every tick's Python half on the GIL, while 8 independent
  processes are what measured 7.54x.
- Prompts go in a sliding window, not all at once — `Engine.submit` allocates
  the recurrent state slot THERE, not at admission, so the state pool bounds
  how many may be in flight. A corpus of thousands exhausts it on line one.

## 2026-08-30 — accept: TF32 for the backward's fp32 matmuls, -10.8%

- `torch.backends.cuda.matmul.allow_tf32` defaults to False and the string
  appeared nowhere in this tree, so every `@` in the eager backward ran on
  SIMT FP32 cores: **308 of 1332 ms**, 23% of a train step, as
  `cutlass_*_simt_sgemm` and `sm80_xmma_gemm_f32f32`.
- Two lines. GPU-busy **1332.4 -> 1189.0 ms**. The estimate beforehand was
  -20%; the measurement is -10.8%, and the measurement is the number.
  [wins/2026-08-29-tf32-train-matmuls.md](docs/experience/wins/2026-08-29-tf32-train-matmuls.md)

## 2026-08-30 — phase exit: the quantization parity gates were modelling kernels nothing runs

- Three gates had been red on CUDA for a while: `linear_fp4`, `linear_fp4_fp8`
  and `linear_fp8` parity. All three picked their reference with `M > 1` when
  the dispatch boundary is `M > _MX` and `_MX` is 8, so each compared a kernel
  against a reference for a DIFFERENT numeric path and the mismatch read as a
  2-4% kernel error. Every kernel was at its floor against the reference it
  actually computes.
- A useful side finding: `linear_fp8` keeps the activation in bf16 through
  M=8, exact to 3e-04. Only above it does the ~2.6% e4m3 activation quant
  apply — the B=1..8 decode range is more accurate than the tests assumed.
  [errors/2026-08-29-parity-gates-modelled-the-wrong-kernel.md](docs/experience/errors/2026-08-29-parity-gates-modelled-the-wrong-kernel.md)
- Same shape in two more places: `test_paged_attention_vs_naive` built its KV
  cache in f32 while the pool is bf16, and the tape gradcheck's bf16 central
  difference disagreed with ITSELF by 128% across step sizes while the tape
  value held to 0.4% across targets. CUDA failures 12 -> 8.

## 2026-08-29 — reject: the fp8 dX kernel (correct, but an 80-block grid)

- Written to remove the last dequantized-weight materialization; it does, the
  op peaks **2.054 -> 0.060 GiB**. But it tiles the OUTPUT [M, K], so on
  lm_head's K=5120 the grid is **80 blocks** on a 132-SM card while the
  contraction runs 7760 serial iterations over N=248320. Measured 0.618x /
  0.549x at 1x64 / 1x128, and only noise-level wins once the grid fills.
- The kernel is correct: norm-relative error **0.0034** at every size, of which
  0.0021-0.0024 is the bf16 casts it inherits. Its parity test failed because
  an elementwise relative gate is wrong for a cancelling output like dX — as
  was my first error metric, which reported 11001 for a 0.3%-accurate result.
- Reverted. Reopen path: a split-N variant, the trick
  `linear_fp4_fp8_decode` already uses.
  [errors/2026-08-29-fp8-bwd-kernel-grid-too-small.md](docs/experience/errors/2026-08-29-fp8-bwd-kernel-grid-too-small.md)

## 2026-08-29 — phase exit: the frozen-weight backward stops materializing lm_head

- Attributing the backward peak per op named a single culprit: one
  `linear_fp8_frozen` call peaked **14.209 GiB** where every other op in the
  model is under 0.12. Its fp8 branch has no tilelang kernel, so it eagerly
  materialized the whole [248320, 5120] weight in f32 — twice, once more to
  fold `oscale` in.
- `oscale` now folds into the [M, N] gradient (where the fp4 kernel path
  already puts it) and the weight is dequantized 512 MiB at a time. The chunk
  step is a multiple of the scale's row granularity — an fp8 scale covers 128
  rows, so a naive row slice silently reads the wrong scale.
- Op peak **14.209 -> 2.054 GiB**, and every LoRA shape got FASTER as well:
  1x64 **50.3 -> 57.7 tok/s, 47.0 -> 31.0 GB**; 2x256 **178.2 -> 194.7,
  76.5 -> 67.7**.
  [wins/2026-08-29-frozen-bwd-chunked.md](docs/experience/wins/2026-08-29-frozen-bwd-chunked.md)

## 2026-08-29 — phase exit: full fine-tuning of the 27B fits one card

- Three arithmetic blockers, settled by adding up the ledger rather than
  profiling: Adam's m+v is **200.4 GiB** against 50.1 GiB of bf16 weights;
  every weight gradient coexisting is another **50.1 GiB**, forced by
  `clip_grad_norm`'s global norm; and `keep_master` held the served bytes
  beside the masters for **14.9 GiB** the forward never reads.
- Adafactor (factored second moment, 0.03 GiB) clips each update itself, so no
  global norm is needed and every gradient is consumed and freed inside
  backward. `drop_quantized()` frees the dead bytes at the training entry
  points — `load_hf` stays a bit-exact round trip.
- Measured B=1 T=64: materialize 50.10, forward 51.80, **backward peak 73.20 of
  95 GiB**, and backward returns to 50.37 — no gradient outlives its update.
  [wins/2026-08-29-full-finetune-fits.md](docs/experience/wins/2026-08-29-full-finetune-fits.md)

## 2026-08-29 — phase exit: the embedding table stops being resident three times

- The gather kernel demanded an f32 table, so the 27B's bf16 [248320, 5120]
  embedding lived as a cached 4.7 GiB f32 copy beside the 2.4 GiB original;
  and the tape computed a dense 4.7 GiB f32 gradient for that table every step
  even though LoRA freezes it. Both are memory AT the peak.
- `make_embedding` gained a bf16 body (CUDA only — the C target cannot codegen
  bfloat16), and `Tape.backward(needs=...)` lets a handler skip an expensive
  gradient for a leaf nobody reads.
- 27B LoRA, all four shapes better on both axes: 1x64 **50.3 -> 51.1 tok/s,
  47.0 -> 42.3 GB**; 1x256 113.5 -> 115.2, 57.5 -> 52.9; 2x256 **178.2 ->
  182.3, 76.5 -> 67.6**.
  [wins/2026-08-29-embedding-f32-copies.md](docs/experience/wins/2026-08-29-embedding-f32-copies.md)

## 2026-08-29 — reject: streaming gradient release is not a memory win

- `Tape.backward` can now hand each gradient to a callback the moment it is
  final, so anything the step does not keep is dropped instead of living until
  backward returns. The mechanism is correct and equivalence-tested; wiring it
  into `train._step` was reverted.
- The 8.9-9.3 GiB it reclaims was measured **after backward returned**, not at
  the peak, and peak is what OOMs. Three of four 27B LoRA shapes regressed
  3-7% (1x128 80.5 -> 75.4, 1x256 113.5 -> 105.9) with peak GB unmoved; only
  2x256 saved anything (76.5 -> 71.8 GB).
  [errors/2026-08-29-streaming-grad-release-no-peak-win.md](docs/experience/errors/2026-08-29-streaming-grad-release-no-peak-win.md)

## 2026-08-29 — default flip: the gated-delta backward is chunked (CHUNK=16)

- `reference.gdn_backward` looped over the time dimension twice in Python, ~28
  launches per step per GDN layer: **491K kernels a step, 62% micro-ops**. The
  adjoint of the chunkwise-WY form replaces both loops, so the sequential
  dimension is `t/16` and `states`/`ps`/`deltas` are gone.
- GPU-busy **2296 -> 1405 ms** (1.63x), kernels **491K -> 158K** (3.1x fewer).
  End to end on the 27B: 1x256 **41.2 -> 113.5 tok/s (2.75x)**, and 2x256 now
  FITS at **178.2** where it used to OOM — best throughput **4.32x**.
- Chunk 16, not the upstream 64, and the reason is precision: same algebra,
  different f32 reduction order. Gated both ways — the adjoint is exact in f64
  (< 1e-12 vs autograd, term by term) and the f32 error is held at the serial
  scan's level (1.3-2.2x of 3-9e-7, against 5-10x at chunk 64).
  [wins/2026-08-29-chunked-gdn-backward.md](docs/experience/wins/2026-08-29-chunked-gdn-backward.md)
- `gdn_chunk_core` collapsed 40 lines -> 8: forward and backward now share one
  implementation of the algebra.

## 2026-08-29 — phase exit: reinforcement learning exists

- The tree had SFT (`train_step`), corpus SFT (`pretrain`) and distillation
  (`opd_loop`) and called it RL training. No reward, no advantage, no policy
  gradient anywhere — `SamplingParams.logprobs` shipped with the comment "what
  a policy gradient needs" and nothing consumed it.
- GRPO lands as `train.grpo_loop` / `rl_step` / `group_advantages`, exposed as
  `tilerl train --rl`. No critic (the group mean is the baseline), so rollout
  and update share one model and one set of weights. It is not a second
  training path: `_step` carries the tape, the frozen-base filter, the
  finite-step rejection and the optimizer, and SFT and RL differ ONLY in the
  logit-gradient function — the policy gradient is the causal-CE gradient
  scaled per row by the advantage.
- Gated algebraically, not just end to end: A=1 must equal an SFT step
  parameter-for-parameter, A=0 must be an exact no-op. Reward on the tiny model
  goes 0.17 -> 1.00 in seven steps through the real engine, and on the **27B**
  (LoRA rank 16, group 4, H20) **0.406 -> 1.000 in five**.
- Two fixes were needed to make that run mean anything: the rollout was
  sampling from an earlier policy (prefix cache + captured graph both cache
  across an update), and adapter-only training carried ~27 GB of bf16 masters
  it never reads, which OOM'd the card at 95.21 GiB (now 59.8).
  [wins/2026-08-29-grpo.md](docs/experience/wins/2026-08-29-grpo.md)
- Training THROUGHPUT is unchanged and its root cause is recorded separately:
  491K kernels a step, 62% of them micro-ops from `gdn_backward`'s
  per-time-step Python loop.
  [errors/2026-08-29-train-step-is-the-gdn-per-step-loop.md](docs/experience/errors/2026-08-29-train-step-is-the-gdn-per-step-loop.md)

## 2026-08-29 — default flip: M-row GEMV for 2 <= M <= 3 decode rows

- `linear_*_mma8` pads M to 8 rows, so a decode of TWO requests cost the same
  tick as eight and was **slower in aggregate than serving one** (71.9 vs 86.6
  tok/s). The GEMV now takes a compile-time row count and reuses each decoded
  weight tile across M rows. Paired A/B in one process, aggregate tok/s:
  **B=2 71.9 -> 111.4 (1.55x)**, B=3 108.2 -> 119.7 (1.11x), B=1/4/8 unchanged.
  The crossover is measured, not assumed: M=4 loses (30.1 vs 27.6 ms replay).
  [wins/2026-08-29-m-row-gemv.md](docs/experience/wins/2026-08-29-m-row-gemv.md)
- Root cause of the speculative tick's unexplained cost, which was attributed
  by subtraction and blamed on the draft head: a draft step is **2.06 ms, not
  ~11**, and the verify replay's +19 ms over plain decode is entirely these two
  linear kernels (GDN contributes +0.08 ms). Five A/B experiments had gone
  looking for a draft-head defect that was never there.
  [errors/2026-08-29-spec-cost-was-the-linear-not-the-draft.md](docs/experience/errors/2026-08-29-spec-cost-was-the-linear-not-the-draft.md)
- Bench harness sweeps `--batches 1,2,4,8`: the loss region sat between the two
  endpoints the old `1,8` sweep measured.

## 2026-08-29 — verdict: speculative decoding REJECTED on throughput, kept on correctness

- Speculation lives in the engine: a decode row drafts off the trunk's last
  hidden and the SAME forward verifies it as a `seq_q = 1+depth` row. The paged
  KV self-heals; the gated-delta state is kept after every chain step and the
  accepted length selects one — no snapshot, no second forward.
  Goodput is real: **1.87 committed tokens per trunk forward, 43-47%
  acceptance**. Throughput is not. A draft disables graph capture, and the
  shipped decode IS graph-captured: **B=1 86.2 tok/s graph vs 17.6 speculating**
  — 4.9x slower than what ships. The first reading here compared against the
  EAGER path (10.9 tok/s), which is not what runs. Retracted the same day;
  speculation pays only once a spec tick can be captured, one graph per
  `(B, 1+depth)`.
  [wins/2026-08-29-spec-decode-net-win.md](docs/experience/wins/2026-08-29-spec-decode-net-win.md)
- It was a 6.5x net LOSS first. The head shipped dense bf16, which
  `Backend.linear` serves at ~30 GB/s — 9.7 ms per projection against the
  trunk's 0.13 ms — so one draft step cost more than the whole 64-layer tick.
  Serving the head as fp8 fixed it, but only after deleting a duplicated
  `materialize` in `build_engine` that re-bound `draft.params` and left the
  head's own `Model` reading the original weights. That one line made three
  consecutive real fixes measure as no-ops.
- The head itself was broken until today: `load_draft` skipped the zero-centered
  RMSNorm +1 fold that `load_hf` applies, and the head emitted logits
  ANTI-correlated with the trunk (its argmax ranked 248191 of 248320).
  Teacher-forced agreement 0.0% -> 84.4%.
  [errors/2026-08-29-draft-head-missing-zero-centered-fold.md](docs/experience/errors/2026-08-29-draft-head-missing-zero-centered-fold.md)

## 2026-08-29 — phase exit: prefill 1.15x, and four false wins retired

- **Prefill 1836 -> 2109.7 tok/s (1.15x)**, from two changes that the
  per-kernel GPU table could not show: the tick's KV descriptors crossed to the
  device 971 times per prefill, synchronously (+9%), and lm_head ran over all
  512 positions of a chunk to use one (+3.1%). MMLU held at 81.0% throughout.
  [wins/2026-08-29-pinned-kv-descriptors.md](docs/experience/wins/2026-08-29-pinned-kv-descriptors.md),
  [wins/2026-08-29-prefill-lm-head-last-token.md](docs/experience/wins/2026-08-29-prefill-lm-head-last-token.md)
- **Rejected on measurement:** speculation (0.43-0.76x of plain graph decode,
  every batch and depth); the chunked GDN prefill (third attempt, broken and
  7.1x slower); prefill graph capture (unblocked, ~1%, reverted — capture pays
  where dispatch is exposed, and prefill's is not).
- **Four claims retired as measurement errors, not results.** Speculation's
  "1.14x" compared against the eager path when the shipped decode is
  graph-captured; "wins at B=8" compared two scripts with different prompts and
  warmups; the sglang prefill gap compared our B=1 against their B=8 (2.03x,
  not 2.5x); and "chunkwise-WY was never measured" missed two A/B tables in the
  same directory. Every number was real and every conclusion was wrong.
- The harness now makes those four mistakes structurally hard: the spec suite
  measures its own no-draft arm, prefill takes three readings, a baseline raise
  requires the row's spread to clear the margin, and `--suite accuracy` gates
  MMLU so a change that breaks the logits cannot pass on speed alone.
- **GDN prefill is SM-limited, measured**: 2x the blocks in one launch costs
  1.67x the time, so a V split is worth 1.20x on the kernel and 5.5-8.6%
  overall. Not implemented — it needs the gated RMSNorm moved to a second
  launch in a file where three rewrites have failed, and it does not reach the
  1.91x that remains.
  [errors/2026-08-29-gdn-48-blocks-78-sms.md](docs/experience/errors/2026-08-29-gdn-48-blocks-78-sms.md)
- `tilerl train --opd` runs the EMA self-teacher loop; `add_lora` attached to
  nothing on a dense base, so the OPD path could not run on CPU at all.

## 2026-08-29 — the harness gates accuracy, not only speed

- Every gate in the bench harness measured tok/s, so a change that broke the
  logits passed all of them. `--suite accuracy` seeds at **81.0%** (162/200,
  0-shot MMLU, fixed greedy slice); `mmlu.py`'s scoring is now shared with it
  rather than living in a script nobody ran on a green bench. Baseline raises
  additionally require the row's spread to be under the raise margin.
- The 27B LoRA train rows pass again after the backward-tile revert
  (26.1 / 32.5 / 37.6 tok/s, 0.983-0.995x) — the earlier 0.962x FAIL was that
  change, not a regression.
- Speculative decoding: the engine's multi-token verify path is CORRECT
  (matches T=1 greedy exactly at chunk widths 2/5/9, and the block loop
  reproduces greedy with an always-wrong and an always-right draft). The draft
  head is one full-attn layer, 1% of the trunk, so speculation costs about 10%
  of a tick. `fc` consumes `concat(embed, hidden)`, not the reverse.
  [errors/2026-08-29-draft-probe-parallel-implementation.md](docs/experience/errors/2026-08-29-draft-probe-parallel-implementation.md)
- Chunkwise-WY GDN prefill was ruled out on 2026-08-24 by an argument, not a
  parity run, and the argument is wrong — the intra-chunk term the objection
  said was dropped is carried by `wy_fast`'s `W`/`U`. A 1.36x prefill win was
  closed off for four days.
  [errors/2026-08-29-chunkwise-wy-wrongly-ruled-out.md](docs/experience/errors/2026-08-29-chunkwise-wy-wrongly-ruled-out.md)

## 2026-08-28 — phase exit: 27B LoRA training is 28-40x faster — the step was launch-bound

- One train_step issued **671,123 kernel launches** (8 layers, 1x64); every
  tileRL kernel was in the double digits and 668K were 1-3 us torch micro-ops
  from `gdn_backward`'s Python loop over (time step, value head). Vectorizing
  both scans over heads: **0.8 -> 26.5 / 32.6 / 38.2 tok/s** at 1x64/128/256,
  20,753 kernels. Two earlier hypotheses (the fp4 dequant; GDN being expensive
  to compute) were both wrong and are recorded as such.
  [wins/2026-08-28-gdn-backward-launch-bound.md](docs/experience/wins/2026-08-28-gdn-backward-launch-bound.md)
- The frozen-base backward is now one fused dequant+GEMM kernel (`gx = grad @ W`
  contracts over the weight's ROW index, so a packed slab is already gemm_nn's
  B tile — no transpose, no materialized weight). Worth 1.18x on its own.
- Bundled NextN draft head loads and drafts. The acceptance rate quoted here
  on 2026-08-28 was void, as were four later readings — the probe
  re-implemented the engine's decode path and diverged from it five different
  ways. No acceptance number stands yet.
  [errors/2026-08-29-draft-probe-parallel-implementation.md](docs/experience/errors/2026-08-29-draft-probe-parallel-implementation.md)

## 2026-08-28 — verdict: mma8 bf16 block scales rejected (flat), KV split count by occupancy accepted

- Splits by occupancy (B=1 grid was 64 blocks on 78 SMs): B=1 decode
  **92.4 / 88.9 / 80.1** at 512/8k/32k, +1.1-1.9%.
  [wins/2026-08-28-decode-split-by-occupancy.md](docs/experience/wins/2026-08-28-decode-split-by-occupancy.md)
- bf16/f16 scales measured 0.998x / 0.991x at B=8 and were reverted.
  [errors/2026-08-28-mma8-bf16-scale-no-gain.md](docs/experience/errors/2026-08-28-mma8-bf16-scale-no-gain.md)

## 2026-08-28 — eval: MMLU 0-shot 76.3% on the 27B NVFP4, and the first 128K/256K rows

- `SamplingParams.allowed_ids` (logit mask over the answer-letter ids) turns a
  chat-tuned model's prose into a scorable answer: 0.8% -> **76.3%**, 0 unparsed.
  The sglang arm is invalid — its bf16 checkpoint emits garbage
  ([errors/2026-08-28-sglang-bf16-checkpoint-garbage.md](docs/experience/errors/2026-08-28-sglang-bf16-checkpoint-garbage.md)),
  so those numbers are throughput only.
  [wins/2026-08-28-mmlu-letter-restricted.md](docs/experience/wins/2026-08-28-mmlu-letter-restricted.md)
- Long-context decode, B=1, one card: **61.1 tok/s at 128K, 48.1 at 256K**
  (64-split decode attention).
  [wins/2026-08-28-split-kv-decode-attention.md](docs/experience/wins/2026-08-28-split-kv-decode-attention.md)

## 2026-08-28 — phase exit: P1 shape — 27B trains on one card (frozen fp4 base + LoRA)

- A tape recording of a quantized linear no longer needs a bf16 master: with no
  master it runs the real fp4/fp8 kernel and yields dX only. That removes 54 GB
  of masters + 216 GB of Adam moments and makes `add_lora` + `train_step(...,
  trainable=...)` the shape P1 asked for; `opd_loop` gains the EMA self-teacher
  (adapters only). Harness `train` suite now has a 27B row.

## 2026-08-28 — feature: thinking effort

- `reasoning_effort` (none/minimal/low/medium/high) maps to a thinking-token
  budget; when it is spent the engine closes the reasoning block itself. The
  end-of-think ids come from the caller, so the engine stays tokenizer-free.

## 2026-08-28 — plan: training / RL roadmap (LoRA-OPD → batch decode → TP → CP → full-param)

- `docs/plan-training-rl.md`: physics of the 8×H20 pod for a 27B, six phases
  with gates and effort, the one decision to take (NCCL via a `comm` seam).

## 2026-08-28 — verdict: vs sglang on the same H20 — B=1 1.68× faster, B=8 0.8×, prefill 0.4×

- sglang (bf16, since it cannot run NVFP4 on Hopper) B=1 54.2 / B=8 387 /
  prefill 2512; sglang online-fp8 39.9 / 266.6 / 4022; tileRL 90.9 / 308.6 /
  1836. Record with method and caveats:
  `docs/experience/2026-08-28-vs-sglang-h20.md`.

## 2026-08-28 — phase exit: decode nearly depth-flat — B=1 90.9 @512, 78.6 @32k; B=8 agg 308.6

- Quiet-host gate over the f16 fp8 mma8 + split-KV attention + fused rmsnorm
  tranche: B=1 d512 87.5 → **90.9**, d2k 79.8 → 87.3, d8k 58.5 → 87.9, d32k
  28.7 → **78.6**; B=8 agg d512 286.7 → 308.6, d2k 302.8, d8k 280.2; prefill
  unchanged; verify 1–3 PASS. Ledger:
  `docs/experience/2026-08-28-decode-52-to-84.md`.

## 2026-08-28 — single-launch parallel rmsnorm on sm90 (2.1 µs vs 4.9 for the split-K pair)

- One block per row, 256-thread strided partials, block-wide allreduce,
  bf16 write. The 08-27 single-launch attempt lost 20% because it reduced
  serially; this one keeps the parallelism. Kernels per 8-layer B=1 tick
  142 → 125; verify 1–3 PASS. Note appended to
  `docs/experience/errors/2026-08-27-fused-rmsnorm-regression.md`.

## 2026-08-28 — split-KV decode attention, GQA group as the M tile — 5.7× at 32k (kernel)

- Pure-decode ticks use `paged_attention_decode` (grid KVSPLIT×Hkv×B, the 4
  query heads of a group as tile rows, static partial workspace) +
  `paged_attention_combine`. Kernel-level at 32k, B=2: 1.451 → 0.256 ms per
  layer, relerr 3.3e-3 vs the dense kernel; verify 1–3 PASS. Harness rows
  pending a quiet host. Entry: `docs/experience/wins/2026-08-28-split-kv-decode-attention.md`.

## 2026-08-28 — fp8 on the f16 tensor path (hardware e4m3 cvt) — B=8 agg 298 tok/s

- `cvt.rn.f16x2.e4m3x2` (1 op/pair, was 7) in the fp8 mma8 kernel: X
  converted bf16→f16 per chunk, f16 mma, f32 accumulate. Parity unchanged;
  B=8 agg d512 286.7 → 298.2, d2k 254.7 → 274.6 (host loaded); verify 1–3
  PASS. The same path in the M=1 fp8 GEMV tile collapsed greedy decode
  (f16x2 accumulate overflows on this model's post-norm activations) and is
  reverted — `docs/experience/errors/2026-08-28-f16-tile-overflow-scale-order.md`.
  Entry: `docs/experience/wins/2026-08-28-mma8-decode-gemm.md`.

## 2026-08-28 — verdict: tensor-core decode GEMM for 2≤M≤8 — B=8 agg 286.7 tok/s (+31%)

- `mma.sync.m16n8k16.bf16` fed straight by the twiddle decode (no re-packing:
  a consistent virtual-k permutation on both operands), one block scale per
  lane per k32 chunk on the B fragment, G=4 chunks of loads in flight per
  warp; fp8 twin. Replaces the padded WGMMA w4a8/fp8 decode paths and the
  scalar batched GEMV (register-bound at 204 regs/lane — measured, deleted).
  B=8 agg d512 219 → 286.7, d2k 216 → 254.7, d8k 171 → 202.2; parity M=8
  1.7e-3; verify 1–3 PASS; M=1 stays on the scalar GEMV (2.2× slower through
  mma8). Entries: `docs/experience/wins/2026-08-28-mma8-decode-gemm.md`,
  `docs/experience/errors/2026-08-28-batched-scalar-gemv-register-bound.md`.

## 2026-08-28 — phase exit: B=1 decode 87.2 tok/s, Arle's 84.5 passed

- **B=1 83.9 → 87.2 tok/s (+3.9%)**, d2k/d8k +4%, verify 1–3 PASS. The conv
  window is double-buffered in the state pool (parity plane, flipped once per
  tick) and shifted inside the fused GDN kernel — the last per-layer
  gather/scatter is gone. Same commit: batched M≤8 fp4/fp8 GEMVs (activation
  rows in registers) replace the padded WGMMA decode paths — B=8 agg 215 → 219
  (d512), 204 → 216 (d2k). Entry: `docs/experience/wins/2026-08-28-conv-window-double-buffer.md`;
  ledger: `docs/experience/2026-08-28-decode-52-to-84.md`.

## 2026-08-28 — residual add in the GEMV epilogue — B=1 83.9 tok/s (99% of Arle's 84.5)

- **B=1 82.0 → 83.9 tok/s (+2.4%)**, verify 1–3 PASS, B=8 flat, prefill −2%
  (inside the 3% gate). The three residual GEMVs (o_proj, down, out_proj)
  write `Res + y·oscale` in f32 on the serving path (`Model._add_via`); the
  tape path keeps `backend.add`. n_partition sweep: 2 ≈ 4 > 8 > 16.
  Entry: `docs/experience/wins/2026-08-28-decode-glue-casts.md`. Method
  record of the whole day: `docs/experience/2026-08-28-decode-52-to-84.md`.
- Day total: 52.6 → 83.9 tok/s B=1, 142.7 → 212.5 agg B=8, prefill 1566 →
  1795 tok/s. Left on the table: the conv-window gather/scatter (needs a
  double-buffered pool), GEMV occupancy (7 blocks/SM), and B=8's WGMMA path.

## 2026-08-28 — verdict: packed-math fp8 GEMV — B=1 82.0 tok/s (97% of Arle)

- **B=1 74.9 → 82.0 tok/s (+9.5%)**, verify 1–3 PASS. ncu in the real model
  showed the fp8 GEMV at 7.3 instr/elem, SM 76–85% busy, DRAM 54%. e4m3→bf16x2
  by bit placement (+2^120 rebias) and `fma.rn.bf16x2`, tiles loading from
  global with program-ordered vector loads: in_proj 44.7 → 34.7 µs, out_proj
  19.1 → 15.2 (ncu). Two dead ends on the way: tilelang locals passed to the
  extern by pointer (local memory), and the eager A/B's ~40 µs CPU floor
  hiding the whole effect. Entry:
  `docs/experience/wins/2026-08-28-fp8-gemv-bf16x2.md`.

## 2026-08-28 — phase exit: decode glue removed — B=1 74.9 tok/s (89% of Arle's 84.5)

- **B=1 61.7 → 74.9 tok/s (+21%), B=8 agg 184.7 → 212.6 (+15%)**, all rows
  raised, greedy text unchanged. Python-level attribution (`profile_glue.py`)
  found ~900 per-tick launches of pure glue: parameters re-cast to f32 every
  call, f32→bf16 casts between kernels, `oscale` as a torch mul, GDN state
  bf16↔f32 round trips + gather/scatter. Fixes: cached parameter casts
  (`_const_f32`), bf16-writing rmsnorm/silu on sm90, `oscale` folded into the
  GEMV epilogues, f32 state pool updated in place by the fused GDN kernel.
  Kernels per 8-layer tick 321 → 192. Entries:
  `docs/experience/wins/2026-08-28-decode-glue-casts.md`,
  `docs/experience/errors/2026-08-28-gdn-inplace-raw-serialized.md`.
- Remaining B=1 tick (13.35 ms): fp8 GEMV ~48% and fp4 GEMV ~32% of GPU time,
  both at 48–63% roofline; the rest is <20%.

## 2026-08-28 — verdict: fp4 GEMV was issue-bound; twiddle decode + bf16x2 FMA shipped

- **B=1 54.2 → 61.7 tok/s (+14%), B=8 agg 142.7 → 184.7 (+29%), prefill
  +13%**, every baseline row raised, greedy text unchanged. ncu showed the
  shuffle-LUT GEMV at 82% issue-busy / 40% DRAM (6.8 instr/elem); scale-dtype
  and split-K were measured dead first. Fix: tilelang's twiddling decode
  (packed bf16x2) + `fma.rn.bf16x2`; sm90 serves fp4 in the twiddled layout
  (`reference.twiddle_fp4`, in-place + flag; `save_hf`/reference untwiddle).
  Entries: `docs/experience/errors/2026-08-27-fp4-gemv-issue-bound-ncu.md`,
  `docs/experience/wins/2026-08-27-fp4-gemv-twiddle-bf16x2.md`.
- Arle gap: 61.7 vs 84.5 (73%). Next: the fp8 GEMV (40% of the B=1 tick)
  with the same packed-math treatment; attention at depth (25 tok/s @32k).

## 2026-08-27 — phase exit: benchmark harness is the perf gate; verdict: decode is GEMV-bound, not launch-bound

- **`tilerl bench --suite decode-kv,prefill,kv-reuse,train`** with a snapshot
  baseline (≥0.97× passes, winners auto-raise). Decode-vs-KV-depth curve on
  the 27B: 51.9/48.2/39.7/23.2 tok/s at 512/2k/8k/32k. Prefix cache confirmed
  on GPU. 27B training on one H20 is pending-remote (fp32 masters = 108 GB).
  Entry: `docs/experience/wins/2026-08-27-bench-harness.md`.
- **Verdict reversed:** in-graph per-kernel profile shows GEMVs are 74% of the
  B=1 tick (fp8 40%, fp4 34%), WGMMA 80% of B=8; launch count is a ~2% lever.
  Entry: `docs/experience/errors/2026-08-27-kernel-count-verdict-refuted.md`.
- **Shipped:** `attn_prep` fusion (+2%) and grouped fp8 GEMV (−8% on the big
  shape): B=1 19.28 → 18.38 ms/tick, **54.4 tok/s** (Arle 84.5). Entry:
  `docs/experience/wins/2026-08-27-attn-prep-fp8-gemv-grouped.md`.

## 2026-08-27 — phase exit: the 27B computes correct logits (zero-centered RMSNorm fixed)

- **The 27B produces correct text for the first time.** Root cause of the
  wrong-logits collapse (check 2, all prompts → junk id 158949): Qwen3.5 uses
  **zero-centered RMSNorm `y = x_normed * (1 + weight)`** (HF weight init is
  zeros), but tileRL applied plain `* weight` on all five non-gated norm sites
  (input/post_attn/q/k/final) — a ~2× per-layer residual blowup. The GDN gated
  norm was already correct (not zero-centered). Fix: fold `+1` at load, `-1` at
  save; zero kernel/hot-path change, 3 lines. Now: "France" → " Paris…",
  Fibonacci → correct code, "17+25" → "42". Checks 1/2/3 PASS; throughput
  unchanged 0.97× — now on a CORRECT model. Found by external-ground-truth
  bisection against HF transformers (`hf_reference.py`/`hf_bisect.py`), not by
  cosine probes (inconclusive) or internal parity (kernel+reference shared the
  bug). Entry: `docs/experience/wins/2026-08-27-zero-centered-rmsnorm.md`,
  `docs/experience/2026-08-27-pod-verification.md`.

## 2026-08-27 — verdict: the fp4 GEMV's load width is measurable, register-resident B is expressible

- **Both answers are "yes, and here is the instrument" — neither is a perf
  number, because neither can be one without a GPU.** `micro_size_k`/`GROUP` are
  now call-time knobs on `make_linear_fp4_gemv`; at the defaults the emitted
  CUDA is byte-identical to HEAD's, so the shipped path did not move. All 9
  combinations lower and index exactly, the WQ load width tracks `micro_size_k`
  alone (8 -> LDG.32, 32 -> LDG.128) and the register footprint tracks
  `micro*GROUP` — making **(32,1) the shipped kernel with one variable moved**:
  same 52 register slots, same 32-shuffle decode batch, same 16 B/thread in
  flight, one LDG.128 where (8,4) issues four LDG.32. The sweep collapses to 6
  arms (`scripts/bench_gemv_micro.py`, two gates, run order in
  `scripts/POD-VERIFY.md`); the stale "micro=16/32 tested worse" note is struck,
  it predates `GROUP` and priced micro=32 only at 4x the register footprint.
  Two silent defects died first: a 32-elem micro-tile spans two block-16 scales
  (57% relative error at full speed), and the obvious `for s in range(nseg)`
  parses as `T.serial` and emits a runtime-indexed register array. Separately,
  Marlin's register-resident dequantized B **is** expressible in TileLang —
  `T.gemm`'s SR variant, lowered without `ldmatrix_b` to the same `mma.sync`
  Marlin issues — priced at 12 KiB/CTA of shared memory on the live w4a8 arm,
  not the 24 KiB of `make_linear_fp4_mma`, which `_CUDA_PLAN` makes unreachable.
  Instrument and caveats: `docs/design-kernels.md`.

## 2026-08-27 — phase exit: adversarial defect audit of the 27B serving path

- **Exit.** 30 findings raised, 1 refuted, 5 duplicates merged — **24 distinct
  defects**, ranked by how silently each fails. Landed same-day: the four
  checkpoint cross-checks `_validate_hf_config` never had (rope_theta,
  partial_rotary_factor, rope_scaling, tie_word_embeddings) plus a derived tie
  check against the shard's own tensors — the class that already voided one
  shipped measurement; and `_prefix_state` snapshots (74.81 MiB each at 27B)
  now die with their `PrefixStore` entry instead of growing without bound.
  Everything else needs the pod: the ceiling is the fp4 GEMVs at 33% of roof,
  and the 24 fixes together are worth ~2.65 ms of the 19.03 ms tick.
  Briefing: `docs/experience/2026-08-27-defect-audit.md`.

## 2026-08-26 — phase exit: native fp4 w4a8 load path

- **Exit.** The NVFP4 checkpoint's bytes are now the serving format. OCP e2m1
  is the one internal fp4 grid (the old `e2m1fn` grid had no zero and forced a
  dequant + block-32 re-pack at load); block `B` is a call-time kernel
  parameter (16 and 32 both live); `.oscale` is a real per-output-row epilogue
  slot, so a per-channel FP8 `weight_scale` is served at 8 bits instead of
  being fp4-packed. The e4m3 dequant target is renormalized by an exact power
  of two — a pre-existing 3.8% weight error, and 50% had the native
  magnitudes gone in unchanged, now 2.3% (e4m3's own requant floor).
  `Backend.materialize` replaces the per-call master fallback with a
  load-time conversion table, and the linear-family dispatch collapsed from 19
  string checks to `_CUDA_PLAN` (175 -> 88 non-blank lines, 49.7%), fixing a
  latent out-of-bounds N pad on the fp4->e4m3 arms. 27B weight memory 65.0 ->
  **20.3 GiB params-only / 25.04 GiB actually resident** — corrected
  2026-08-27: the original 20.3 counted `model.params` and missed
  `Backend._embed_table_f32`'s permanent 4.736 GiB f32 copy of the embedding
  table (audit M2; fix is a bf16 `Table` in the sm90 kernel cell, not done).
  Pod confirmation pending-remote. Suite 90 -> 96 passed / 4 skipped.
  Entry: `docs/experience/wins/2026-08-26-native-fp4-w4a8.md`.

## 2026-08-26 — default flip: `load_hf` / `build_random` `keep_master=False`

- **Flip.** The bf16 master is a training-only artifact. `tilerl train` /
  `pretrain` pass `keep_master=True` and the master is regenerated from the
  served quantized bytes (so the STE master matches the kernel exactly);
  `tilerl serve` / `bench` take the default and ship no master to the device.
  `save_hf` now raises without masters instead of writing a shard with no
  linear weights. Entry:
  `docs/experience/wins/2026-08-26-native-fp4-w4a8.md`.

## 2026-08-26 — accept-or-reject: 8-way K-split for small-M fp4 decode — ACCEPTED (+7.5% B=8 aggregate)

- **Verdict.** The decode tick (M<=16) ran the prefill kernel's 2-way
  K-split; at bM=16 a block is 2 warps, so the split's resident warps hide
  HBM latency and the f32 atomics cost less than the occupancy they buy.
  Same-process A/B on the slice4 decode graph (30-tick avg, control =
  shipped ks2): B=2 +8.8%, B=4 +8.1%, B=8 +7.5%, B=1 neutral (M=1 path
  untouched). fro-relerr vs shipped ~1e-7 (atomic reordering); greedy
  tokens flip on near-ties at B=4/8, below the shipped path's own 2% e4m3
  quant noise. Sweep: ks1 -10.8% (hypothesis backwards — the split is
  occupancy, not atomics), ks4 +5.1%, ks8 +7.5%; bf16-A WGMMA rejected
  (-13%, gate-fail). Entry:
  `docs/experience/wins/2026-08-26-batch-decode-h2.md`.

## 2026-08-26 — accept-or-reject: batched-decode arms (shared-X small-M GEMV + k_split=1 WGMMA) — REJECTED (smallm 2.18x slower, ks1 -2.6%)

- **Verdict.** Both B=2..8 decode levers lost the A/B (slice4 graph, B=8,
  same process, 30-tick avg). (1) Shared-X small-M GEMV: 9.76 vs 4.48
  ms/tick (2.18x slower) — the shared-X fix removed the 31x X-traffic
  problem, but the kernel is FMA-issue-bound (8 scalar FMAs/W elem, 85
  inst/micro-tile) vs the WGMMA path's tensor cores; the X reload was
  never the binding constraint. Settles the previous arm's open question:
  a shared-mem-X GEMV does not beat WGMMA at M=16. (2) k_split=1 WGMMA:
  -9.4% all-N, -2.6% large-N-only — k_split=2's atomics are not pure cost;
  the split doubles per-tile dequant parallelism (two SMs on the same
  tile's K-range), load-bearing for the fp4 WGMMA path. Both correct
  (ks1 bit-identical to shipped, fro-relerr 0.0, tokens identical); B=1
  neutral. Reverted (no half-states). The K-split lever itself was not
  dead — the winning move was ks8, not ks1: see the ACCEPTED entry above.
  Entry: `docs/experience/errors/2026-08-26-batch-decode-arms-rejected.md`.

## 2026-08-26 — accept-or-reject: fp4 GEMV dequant issue throughput — REJECTED (PRMT 0.85x, MMA 0.50x; prmt.b32 compiler bug)

- **Verdict.** Replacing the shipped shuffle-LUT dequant with `__byte_perm`
  (PRMT) or MMA (tensor-core) does not beat the shipped GEMV at M=1. Matrix
  A/B on lm_head (N=248320, K=5120): PRMT scalar 0.6128 ms vs shipped 0.5215
  (0.851x, 15% slower — 12 prmt/8 elems vs 8 shuffles/8 elems); MMA at
  block_M=16 1.0353 ms (0.504x, 16x M-waste). The shuffle LUT is already at
  1 op/elem (the floor) and overlaps with the FMA chain — the kernel is at
  55.4% roof, within 3% of the 57% nodecode floor. Secondary finding:
  `prmt.b32` on CUDA 12.9 / sm_90 silently truncates the 32-bit selector to
  16 bits (standalone CUDA repro, no tilelang) — selectors with non-zero
  upper 16 bits produce wrong results. Entry:
  `docs/experience/errors/2026-08-26-fp4-gemv-dequant-issue-rejected.md`.

## 2026-08-26 — accept-or-reject: small-M GEMV for B=2..8 decode — REJECTED (1.56-1.61x slower than the padded WGMMA path)

- **Verdict.** Generalized the fp4/bf16/fp8 GEMV from M=1 to fixed M=8
  (stream W once, M-way FMA) and routed 2<=M<=8 decode through it behind
  `_SMALLM_GEMV`. H20 slice4 graph, same process: B=2/4/8 candidate
  6.33/6.56/7.06 ms/tick vs shipped 3.93/4.11/4.52 — 1.56-1.61x slower.
  Root cause: every warp reloads the full 8-row activation (19.9 GB X
  traffic for lm_head vs the WGMMA path's 636 MB — 31x), plus 8 scalar
  FMAs per W elem vs tensor cores. The kernel is correct (bit-identical
  to the shipping M=1 GEMV row-by-row); the harness's f32-parity flags
  are a gate artifact both shipping GEMVs trip at model-scale K. Kernel
  reverted (impl at 26e6471); the A/B harness stays as dev tooling.
  Shipped from the arm: the `write_tokens` packed-ABI fix (bf16 fused-qkv
  views crashed at B>=2). Entry:
  `docs/experience/errors/2026-08-26-smallm-gemv-decode-rejected.md`.

## 2026-08-26 — phase exit: Qwen3.8-27B full-model serving baseline (52.6 tok/s decode, 1773 tok/s prefill)

- **Baseline.** First full-27B measurement on the real NVFP4 checkpoint
  (`/data00/Qwen3.8-27B-NVFP4`), serving build (fuse_projections, decode graph
  on, sm90), H20: decode B=1 19.03 ms/tick (52.6 tok/s), prefill 1947/1847/1773
  tok/s at 512/2048/8192 (chunked 512). Two load-path defects fixed to get
  there: `qwen38_27b()` GDN value heads 32→48 (checkpoint's A_log is [48]),
  and per-channel `[N,1]` FP8 weight_scale support in load_hf (dequant to
  bf16, repack fp4 — the block-128 native-fp8 path can't express per-channel).
  The loaded model is all-fp4; the native-fp8 path serves the separate
  block-128-scale Qwen3.8-27B-FP8 checkpoint. Entry:
  `docs/experience/wins/2026-08-26-qwen38-27b-baseline.md`.

## 2026-08-26 — fix: P1 audit batch (engine/KV/server correctness, quantized training, ops isolation)

- **Fixes.** Prefix published at seq_len-1 (was one token ahead — a cache hit
  silently skipped it); KV eviction under block pressure on submit and decode
  growth; submit alloc rollback + request failures surfaced via poll/take; EOS
  stop tokens + finish_reason by cause; tokenizer fails closed on a configured
  source; quantized training computes with the bf16 master (AdamW updated the
  master, the fp4/fp8 forward used stale weights); stable CE; Tape reuse guard;
  torch math above ops/ moved behind backend calls. Deploy: devel image (JIT
  needs nvcc), tiny default CMD, pod.yaml sets the 27B source. Bench scripts
  fail closed on busy GPUs and clean the remote dir before sync. 90 passed.
## 2026-08-26 — accept-or-reject: block_M sweep under k_split for fp4 prefill GEMM — REJECTED (shipped bM=128 wins geo-mean)

- **Verdict.** Swept block_M in {64,128,256} x k_split in {1,2} at the 5
  prefill shapes (M=512), H20 pod GPU 7, mean of 20, zeroing inside the
  timed region for ks=2 arms: shipped bM=128/k_split=2 wins the geo-mean —
  next best bM=128 ks=1 at 0.930x. bM=128 is best at every shape under both
  ks values (bM=64 0.74-0.76x: per-block K work halves; bM=256 0.75-0.87x:
  occupancy starved). ks=1 wins only gate/up (1.023x, the saturated-shape
  atomic cost) — ~0.5% geo-mean for a shape branch, not justified. Relerr
  4.0e-3..8.6e-3 vs shipped (reduction order only); ks=2 arms bit-identical.
  Dev-only bench script deleted; the entry's table is the tile-space record.
  Entry: `docs/experience/errors/2026-08-26-bm-sweep-rejected.md`.
## 2026-08-26 — accept-or-reject: stream-K tile scheduling for fp4 prefill GEMM — REJECTED (0.549x geo-mean, occupancy wall)

- **Verdict.** Stream-K port (`examples/gemm_streamk` on the dequant+fp8-WGMMA
  body: 78 blocks partition the first tiles' K-iteration space, then full
  tiles) vs shipped k_split=2, 5 prefill shapes, H20 pod GPU 6, mean of 20:
  geo-mean 0.549x — ~1.8x slower at every shape (0.512x..0.582x), correctness
  green (rel-err 4.0e-3..8.6e-3, split reduction order). The 1-wave grid is
  1 block/SM = 128 resident threads/SM; the body is occupancy-bound, so the
  pipeline stalls with nothing to switch to. Stream-K targets under-filled
  grids (tiles ≤ SMs); these shapes are 4-14 waves, where k_split=2's extra
  blocks are the right lever. Kernel + planner reverted; registry never
  wired. Entry: `docs/experience/errors/2026-08-26-streamk-prefill-rejected.md`.

## 2026-08-26 — accept-or-reject: e5m2 activation precision for fp4 prefill GEMM — REJECTED (0.999x tie, relerr 7.6e-2 vs 1e-2 gate)

- **Verdict.** A=e5m2 arm (W dequant e2m1→e5m2 too — wgmma needs matching fp8
  operands) vs shipped A=e4m3, 5 prefill shapes, H20 pod GPU 6, mean of 20:
  geo-mean 0.999x (tie → reject) and relerr 7.44e-2..8.03e-2 vs the bf16
  oracle (7.6x over the gate; e4m3 floor 3.77e-2..4.03e-2 — e5m2 ~doubles the
  error). Same byte count, same WGMMA instructions → timing ties
  structurally; e5m2's 2 mantissa bits buy nothing. Kernel variants reverted;
  registry never wired. Entry:
  `docs/experience/errors/2026-08-26-e5m2-prefill-rejected.md`.
## 2026-08-26 — accept-or-reject: A=bf16 prefill GEMM (no activation quant) — REJECTED (1.55x slower; W4 error only 1.7e-3)

- **Verdict.** The bf16 arm (`make_linear_fp4_mma`: A stays bf16, W dequant
  e2m1→bf16, bf16 WGMMA — the pre-fp8 prefill kernel, still the registered
  `linear_fp4` fallback) A/B'd against shipped A=e4m3 + fp8 WGMMA k_split=2
  at the 5 prefill shapes (M=512), H20 pod GPU 7, mean of 20, arm A timed
  end-to-end (quant + zeroing + split kernel): geo-mean B/A 0.646x
  (0.624-0.671 per shape) — 1.55x slower. H20 fp8 WGMMA has 2x bf16
  throughput and the GEMM is compute-bound. Accuracy decomposition vs the
  torch f32 oracle: bf16 arm 1.7e-3 (pure W4 error, 24x under gate), shipped
  e4m3 ~4e-2 (A-quant + weight requant dominate) — the accuracy A=bf16 buys
  is worthless, since W4 is not the bottleneck and the model tolerates the
  e4m3 ~4%. Shipped path unchanged. Entry:
  `docs/experience/errors/2026-08-26-abf16-prefill-rejected.md`.

## 2026-08-26 — accept-or-reject: 2-way K-split for fp8 prefill GEMM — ACCEPTED (+7.4% geo-mean, zeroing included)

- **Verdict.** `make_linear_fp4_fp8_mma` gains a `k_split` param; k_split=2
  (3D grid, f32 atomic add into a zeroed output) is the sm90 default. The
  sweep's +8% geo-mean excluded the output zeroing; the A/B
  (`scripts/bench_fp8_split2.py`, H20 pod GPU 6, mean of 20) includes it:
  geo-mean 1.074x (down 1.181x, out 1.136x, z 1.049x, qkv 1.042x, gate/up
  0.974x — already 4+ waves). Rel-err 4.0e-3..8.6e-3 vs the shipped kernel
  (same fp8 math, split reduction order). `uv run pytest -q`: 75 passed,
  4 skipped. Entry: `docs/experience/wins/2026-08-26-fp8-prefill-split2.md`.
## 2026-08-26 — accept-or-reject: native-fp8 in_proj_qkv+z fusion (qkvz) for prefill — ACCEPTED (1.17x, 0.2572 -> 0.2204 ms)

- **Verdict.** Each GDN layer projects the same post-norm hidden with
  `in_proj_qkv` (h→10240) and `in_proj_z` (h→6144), both native-fp8 in the
  Qwen3.8 checkpoint — two `linear_fp8` launches (two activation quants,
  two GEMMs) where one suffices. Extended the fp4 projection-fusion
  mechanism to native-fp8: `_projection_groups` gains `qkvz`,
  `_fuse_projections` concats `.w8`/`.wscale` along N (lossless: 10240 = 80
  blocks, 6144 = 48) plus the bf16 master the CPU/decode path computes
  with, and `_gdn` splits the fused output at `cfg.linear_qkv_dim`.
  Same-process A/B at prefill shapes (M=512, K=2048, N=10240+6144) on the
  H20 pod: 0.2572 → 0.2204 ms (1.17x), relerr 0.0 (bit-identical — same
  per-output dot products, one launch + one activation quant). Serving-only
  via `fuse_projections`; parity on CPU in
  `tests/test_fused_projections_parity.py`. Entry:
  `docs/experience/wins/2026-08-26-fp8-qkvz-fusion-prefill.md`.

## 2026-08-25 — accept-or-reject: chunked GDR scan for GDN prefill — REJECTED (0.90x, bf16 precision wall)

- **Verdict.** The FlashQLA chunk-WY pipeline (6 kernels: cumsum, kkt, solve,
  recompute, state, o) was ported from agent-infer and wired as the default
  for the 6-arg scan and full-GDN prefill on sm90. A/B at slice4
  prefill-512 shapes: serial mega-kernel 4.38ms vs chunked 4.88ms (0.90x),
  and the chunked path has 26% output error at scale=1.0 inputs (serial is
  exact). Root cause: WY is O(T·C·K) vs serial's O(T·K²) — the serial
  kernel is compute-bound with state in L2, so the chunked pipeline's extra
  FMAs cost more than the state-traffic savings; bf16 intermediates between
  pipeline stages lose precision at realistic input scales. Reverted both
  wirings (serial is the default again); removed the GDR kernels after
  rejection (keep condition — state > L2 or few heads — unreachable for
  Qwen3.8; re-port from agent-infer if a state>L2 model appears); kept the
  test/docstring fixes. Entry:
  `docs/experience/errors/2026-08-25-gdn-chunked-gdr-rejected.md`.

## 2026-08-25 — accept-or-reject: GDN chunk NewState write hoisted out of the token loop — ACCEPTED (14.8% kernel, 1.993 -> 1.699 ms)

- **Verdict.** The fused GDN chunk-prefill kernel wrote the full 128-float
  state column to global `NewState` every token, but the caller only consumes
  the chunk-end state (next chunk's seed) — ~1.6 GB/layer of dead traffic at
  T=512. Write once after the scan; the recurrence was already
  register-resident. Same-process A/B on slice4 prefill-512 shapes: 1.993 ->
  1.699 ms, out + new_state allclose. Decode rows (T=1) unchanged. Entry:
  `docs/experience/wins/2026-08-25-gdn-chunk-state-writeback.md`.

## 2026-08-25 — phase exit: decode graph covers B>1 (per-batch-size bucket capture)

- **Shipped.** `_DecodeGraph` now replays per batch-size bucket, captured
  lazily on the first pure-decode tick of each size — B>1 no longer pays the
  eager fallback's ~5ms/tick fixed launch tax (~900 Python dispatches). Mixed
  ticks (decode + prefill chunk) still run eager. Slice4, idle H20 window:
  graph beats eager 3.0x at B=1, 1.4x at B=8; B=8 aggregate 1128 tok/s
  (slice), ~101 tok/s extrapolated to the full 27B — across the 80 target.
  Entry: `docs/experience/wins/2026-08-25-decode-graph-batch-buckets.md`.

## 2026-08-25 — accept-or-reject: fp8 prefill GEMM N-tile sweep — v_n64 ACCEPTED (+33% geo-mean), v_int32/v_ws/v_sota/v_m64 REJECTED

- **Verdict.** The fp4->e4m3 dequant + fp8 WGMMA prefill kernel
  (`make_linear_fp4_fp8_mma`) was neutral vs bf16 at large K (down 1.03x,
  out 1.01x) because the 128 N-tile left small-N grids under 1 wave, so the
  dequant and WGMMA phases aligned across resident blocks and the tensor
  cores idled. Sweep of 6 variants (`scripts/_sweep_fp8_prefill.py`, idle
  H20): N-tile 128->64 doubles the grid (2+ waves on every shape) for +33%
  geo-mean TFLOP/s with exact parity (rel-base 0.00). Shipped kernel via
  `backend.linear_fp4`: gate/up 175.8->209.5, down 113.8->157.5, qkv
  143.4->183.4, z 127.3->175.0, out 103.8->150.5 TFLOP/s; the down/out
  shapes that were neutral vs bf16 are now 1.40-1.42x. Rejected: v_int32
  (block_K=32 + accumulator scaling, 2x slower), v_ws (warp spec on,
  slower), v_sota (256x128 tile, slower), v_m64 (block_M=64, neutral).
  v_n64_split2 (+8% more, needs a zeroed-output wrapper) is the follow-up.
  Entry: `docs/experience/wins/2026-08-25-fp8-prefill-n64-tile.md`.

## 2026-08-25 — phase exit: engine scheduler → SOTA continuous batching with chunked prefill (mixed prefill+decode forwards)

- **Shipped.** Replaced the serial decode-first scheduler with vLLM/sglang
  continuous batching mirrored through agent-infer's `build_forward_plan`:
  waiting/running queues, per-tick token budget
  (`StepLimits.max_num_batched_tokens`, default 512), decode rows first plus
  at most one prefill chunk sharing one mixed forward (no preemption day-1).
  Per-row `seq_q_lens` threads through paged_attention, write_tokens, and the
  GDN chunk kernel so a mixed batch (decode rows + a prefill chunk) runs one
  forward. Also fixed a per-tick tilelang recompile loop: `_make_kv` now
  allocates the block_table at fixed width (pool `num_blocks`) instead of
  `max(len(r.blocks))`, which baked a new `Mb` compile const on every block
  growth (5 recompiles / 30 decode ticks before). Mixed batches exposed a
  latent MMA warp-partition crash (bM=48/80/96/112 do not compile under
  Square policy); the sm90 linear paths now snap bM to {16,32,64,128}.
  Entry: `docs/experience/wins/2026-08-25-engine-scheduler-batch.md`.

## 2026-08-25 — accept-or-reject: GDN chunk kernel local state column + bf16 IO — ACCEPTED (21.6% faster, 2.32 -> 1.82 ms)

- **Verdict.** The serial GDN chunk prefill kernel was load-latency-bound
  (strided state column streamed through global 4x/token). Carrying the
  128-float state column in a per-thread `T.alloc_local` array + 4-accumulator
  unrolling (ILP hides L1 latency) + bf16 IO on Q/K/V/Z/Window halves the
  load traffic: 2.32 -> 1.82 ms (21.6%) on the H20 quiet-window sweep.
  8-acc and fused-dot reassociation tested worse. Parity green (4/4 GDN
  tests, rel-err 2.7e-3). Entry:
  `docs/experience/wins/2026-08-25-gdn-chunk-local-state-bf16.md`.

## 2026-08-25 — accept-or-reject: dual-format fp8 weights for prefill — REJECTED (fp8 MMA already wired)

- **Verdict.** Proposed packing an e4m3 copy of every fp4 projection so
  prefill could use fp8 tensor cores. Killed by one bench: `backend.py:431`
  already routes `linear_fp4` through `make_linear_fp4_fp8_mma` on CUDA M>1 —
  prefill already computes on fp8 (176 TFLOP/s on gate/up, 1.46x over the
  bf16 fallback). The prefill gap is kernel efficiency, not format: the fp8
  MMA sits at 59% of peak and goes neutral at large K (down/out ~1.03x).
  Entry: `docs/experience/errors/2026-08-25-dual-format-fp8-rejected.md`.

## 2026-08-25 — default flip: serving fuses same-input fp4 projections (qkv/ab/gate_up) — decode +4.8%

- **Flip.** `cmd_serve` now loads with `fuse_projections=True`: same-input
  fp4 projections concat losslessly along N at load (per-32-block scales are
  per-row) and one GEMV replaces the group's launches. Training keeps the
  unfused masters (the fused key has none — its tape backward would have
  nowhere to land the STE grad). A/B on slice4, decode graph: 1.821 → 1.734
  ms/tick (549 → 577 tok/s slice); prefill +1.3% (noise). The win is
  per-kernel replay overhead, not Python dispatch (the graph already
  amortizes that) — the eager B>1 path stands to gain more. Entry:
  `docs/experience/wins/2026-08-25-projection-fusion-decode.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV shared-memory dequant ping-pong — REJECTED (2.5-3.3x slower than register group4)

- **Verdict.** Tried to move the fp4 GEMV dequant off the FMA critical path
  via shared memory (the register double-buffer spilled in round 6): a
  same-warp ping-pong (decode g+1 -> shared, FMA g from shared) and a
  producer/consumer warp split (threadIdx.z role, RING=3 SPSC ring, producer
  2 groups ahead, consumer issues zero shuffles). Both bit-exact vs the
  shipped kernel (rel-err 2.74e-3), both 2.5-3.3x slower (14-17% roof vs
  group4's 43%, contended H20): the STS+LDS round-trip and per-group
  `bar.sync` cost more than the shuffle issue they remove, and the LDS
  latency lands on the critical path. group8 ties (same issue/elem, more
  regs). bf16 IO was already done in round 1. The register group4 stays; the
  46%->57% gap to the nodecode floor needs fewer dequant instructions/elem,
  not a different buffer. Entry:
  `docs/experience/errors/2026-08-25-gemv-shared-pingpong-rejected.md`.

## 2026-08-25 — phase exit: final 80/3800 bench — decode 49 / prefill 1172 tok/s, not met; gap is dequant issue throughput + GDN chunk, not physics

- **Verdict.** Final measurement at HEAD ea8ba7f (f32 scales, grouped dequant)
  in a fully-idle H20 window (all 8 GPUs at 0%, BW 3312). Slice4 (3 GDN + 1 FA,
  the 27B's exact 3:1 mix), graph-captured: 1.828 ms/tick decode, 0.0557 ms/tok
  prefill-512. Extrapolated with lm_head (0.5195 ms, 55.4% roof) and fixed cost
  counted once: decode 20.41 ms/tok (49.0 tok/s) vs the 80 target (1.63x gap),
  prefill 0.853 ms/tok (1172 tok/s) vs 3800 (3.2x). Rooflines: decode 162-196
  tok/s (20.4 GB at 3.3-4.0 TB/s), prefill 5898 tok/s (25.7 TFLOP at 296
  TFLOPS) — both targets are 41-64% of roof, physics allows them. The decode
  gap is the fp4 GEMV dequant issue throughput (direct kernel 46% roof, lm_head
  55% — the bf16 GEMV hits 42-116%, so the dequant is the cap); the prefill gap
  is the GDN serial scan (27.4% of the tick, WY rejected 2.6x). Entry:
  `docs/experience/wins/2026-08-25-gemv-instr-gdn-wy.md`.

## 2026-08-25 — accept-or-reject: fp4 packed scales f32 -> e4m3 — REVERTED (5-11% decode regression, not neutral)

- **Verdict.** The internal fp4 per-32-block scale changed from f32 to e4m3fn
  bytes (uint8 view) — the checkpoint's native scale dtype, 4x less scale
  traffic. `pack_fp4` rounds block_max/6 to e4m3; the sm90 fp4 kernels + CPU
  kernel decode in-register (integer bit-trick, no exp2). CUDA parity 31/31
  green. Per-linear decode GEMV is 7-11% slower (issue-bound: the decode
  instructions cost more than the 29% traffic savings), and the regression
  surfaces end-to-end: slice4 decode 1.828 -> 1.937 ms/tick (+6.1%, lm_head
  alone is 28% of the wall at +11%). An earlier "neutral" measurement
  (1.841 ms) was an anomaly, inconsistent with the per-linear data. Prefill
  neutral (compute-bound). Reverted in ea8ba7f; the e4m3 work stays in git
  history. Entry: `docs/experience/wins/2026-08-25-fp4-e4m3-block-scales.md`.

## 2026-08-25 — accept-or-reject: chunkwise-WY GDN prefill — REJECTED (2.6x slower than serial scan)

- **Verdict.** Ported the tilelang branch's chunkwise-WY prefill pair
  (`qwen36_prefill_wy.py` + `qwen36_prefill_scan_o.py`, unmerged
  `feat/qwen36-gdn-megakernel`) to replace the serial GDN prefill chunk kernel
  (27.6% of the prefill-512 tick). Correctness green (parity vs
  `reference.gdn_forward`, rtol=1e-2; T=1 cross-check vs decode kernel) but
  **2.6x slower** on the H20 pod: serial 4.73 ms vs WY 12.38 ms (A=1.62 +
  B=10.76). Root cause: WY is O(T*C*K) (C=64, 64x more FMAs) vs the serial
  O(T*K), and the serial kernel is compute-bound (64KB state fits in L2, 48
  heads saturate the SMs) — the WY's chunk-parallelism has nothing to repay
  its extra compute and two-launch/24MB-intermediate overhead. Reverted to the
  serial kernel. The WY solve math is sound (decay-first delta rule, verified
  1e-17) and the port is in git history if a memory-bound or few-head config
  appears. Entry: `docs/experience/errors/2026-08-25-gdn-prefill-wy-rejected.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV grouped dequant — direct-call 42% -> 46% roof, slice4 decode 1.887 -> 1.837 ms/tick

- **Verdict.** The dequant is grouped 4 micro-tiles at a time: load 4, decode
  all 4 (32 shuffles), then FMA all 4 — hoisting every shuffle off the FMA
  critical path so its latency hides behind the FMA dependency chain.
  Direct-call roofline +9.5-10.9% on both shape orientations (42.0 -> 46.0%,
  38.5 -> 42.7%); slice4 decode 1.887 -> 1.837 ms/tick (530 -> 545 tok/s).
  Rejected: register double-buffer (spills, 22%), 6-op bitcast (32%), 256-entry
  byte-LUT (26%, gather loads), no-X-buffer (19%), 2 accumulators (no help).
  Entry: `docs/experience/wins/2026-08-25-fp4-gemv-grouped-dequant.md`.

## 2026-08-25 — accept-or-reject: fp4 GEMV vectorized dequant — slice4 decode 6.94 -> 1.89 ms/tick (3.67x), big projections 24-33% -> 30-54% roof

- **Verdict.** The fp4 GEMV dequant (the gap named in the 80/3800 verdict
  above) is vectorized: warp-shuffle LUT decode (1 op/elem vs 9 for the
  bitcast) + partial-scale (`acc += s * sum(X*w)`, 1 FP op/elem vs the 2-mul
  chain). The nodecode floor jumped 30% -> 57% roof; the shipped lutshfl
  reaches 44% (78% of floor). Slice2 decode 1.922 -> 1.285 ms/tick (1.50x),
  slice4 6.941 -> 1.893 (3.67x vs WGMMA). The big projections (17408x5120
  class) are capped at ~30% roof by the backend dispatch overhead (~0.022
  ms/call), not the kernel — the direct kernel is at 44%. lm_head (large N,
  overhead amortized) hits 54%. Entry:
  `docs/experience/wins/2026-08-25-fp4-gemv-vectorized-dequant.md`.

## 2026-08-25 — accept-or-reject: final 80/3800 bench — not met (35.5 decode / 992 prefill tok/s), gap is fp4 dequant efficiency, not physics

- **Verdict.** Final measurement at HEAD c97f79c after the bf16 GEMV and
  native FP8 levers, in a fully-idle H20 window (all 8 GPUs at 0%). Slice4
  (3 GDN + 1 full-attn, the 27B's exact 3:1 mix), graph-captured:
  2.662 ms/tick decode, 0.0654 ms/tok prefill-512. Extrapolated with lm_head
  (0.890 ms, measured) and fixed cost counted once: decode 28.2 ms/tok
  (35.5 tok/s) vs the 80 target (2.25x gap), prefill 1.008 ms/tok
  (992 tok/s) vs 3800 (3.8x). The 2026-08-24 all-levers entry's 16.9 tok/s
  decode was an extrapolation artifact (GDN-only per-layer x 64 overcounts
  lm_head 32x); the corrected method on its own data gives 32.9 tok/s — the
  kernel delta since is ~8% (fp8 GEMV decode). Roofline: decode 162-196
  tok/s (20.4 GB at 3.3-4.0 TB/s), prefill 5898 tok/s (25.7 TFLOP at 296
  TFLOPS) — both targets are 41-64% of roof, physics allows them. The gap
  is code: the fp4 GEMV dequant stage caps decode at 24-32% of roof (the
  bf16 GEMV on the same schedule hits 42-116%), and the fp4 MLP prefill
  path runs at 21% of peak (62% of the tick). Eager per-op event spans
  overcount the M=1 tick 3.6x — the graph wall is the metric. Entry:
  `docs/experience/wins/2026-08-25-bf16-gemv-fp8-weights.md`.

## 2026-08-25 — accept-or-reject: native FP8 weights shipped — GDN prefill 1.48x, decode 1.05x, but the 3800 target needs the MLP lever too

- **Verdict.** `load_hf` now keeps the checkpoint's FP8 GDN projections native
  (`<key>.w8` e4m3 + `<key>.wscale` per-128-block, bf16 master recording-only;
  `cfg.fp4` packing skips them) instead of dequantizing to bf16 and re-packing
  to fp4. sm90 dispatch: `make_linear_fp8_mma` (deepgemm 2xAcc, per-128-block
  accumulator scaling, no K-loop dequant) for M>1 prefill, `make_linear_fp8_gemv`
  for M=1 decode. Parity green (local 72, pod CUDA 31). Isolated GDN bench
  (same-process back-to-back, contention-independent): fp4 5.657 ms (62.6
  TFLOPS) → fp8 3.819 ms (92.8 TFLOPS), **1.48x prefill**; decode graph
  2.799 → 2.672 ms/tick, **1.05x**. The 3800 tok/s target is not met by this
  lever alone: the GDN is ~23% of prefill FLOPs, the MLP (NVFP4, unchanged fp4
  path) is ~77% — its dequant efficiency is the remaining lever. Entry:
  `docs/experience/wins/2026-08-25-native-fp8-weights.md`.

## 2026-08-25 — accept-or-reject: bf16 GEMV shipped, but it is NOT the 27B decode lever (premise correction)

- **Verdict.** `make_linear_bf16_gemv` shipped (sm90, parity green local +
  pod CUDA): the fp4 GEMV schedule minus the dequant, 42-116% of HBM roof on
  the bf16 projection shapes vs 20-42% for the padded-M=16 WGMMA it replaces
  (1.9-3.9x). But the SOTA-all-levers bench's claim that "GDN projections are
  bf16, 73% of decode bytes" was wrong: on `cfg.fp4=True` `load_hf` packs
  EVERY projection (GDN `in_proj_*`/`out_proj`, MLP, `lm_head`) to fp4, and
  the decode per-op profile shows only `linear_fp4` (86% of GPU), zero bf16
  `linear` ops. Slice decode before/after: 1.932 → 1.922 ms/tick (noise). The
  bf16 GEMV serves non-fp4 models, not the 27B. The real 27B decode lever is
  fp4 GEMV efficiency (24-33% roof — the dequant stage, since the bf16 GEMV
  on the same schedule hits 42-116%). Entry:
  `docs/experience/wins/2026-08-25-bf16-gemv-decode.md`.

## 2026-08-24 — phase exit: SOTA kernel round complete — 80/3800 not met, gap is kernel efficiency not physics

- **Exit verdict.** Final bench on real NVFP4 slices after all levers (fp4
  GEMV decode, fused GDN decode + chunk prefill, multi-block norm/act, graph
  capture, fp8 prefill WGMMA, FlashAttention paged attn). Slice4 (3 GDN + 1
  full-attn, the 27B's exact 3:1 layer mix) extrapolates to the full model
  with no mix correction: decode 59.3 ms/tok (16.9 tok/s) vs 80 target,
  prefill 1.02 ms/tok (976 tok/s) vs 3800 — 4.7x / 3.9x off under a 99%-util
  co-tenant (high-contention phase: ~15 / ~487 tok/s). Roofline says the
  targets are physics-allowed: decode roof 129 tok/s (30.9 GB at 4 TB/s,
  target = 62% of BW roof), prefill roof 3835 tok/s mixed-dtype / 5317
  fp8-everything — the 3800 target IS the mixed-dtype roofline at 100%
  tensor utilization, unreachable while `load_hf` dequantizes the
  checkpoint's FP8 GDN projections to bf16. The gap is code: bf16 M=1
  projections at ~10-15% of BW roof (73% of decode bytes) need a bf16 GEMV
  path; prefill needs fp8 weight retention plus in-engine fp8 GEMM efficiency
  (16-22% of peak contended, 60-80% isolated). Contention finding: eager
  dispatch is pathological on a shared GPU (784 us/op queue wait, 27.2 ms
  wall vs 2.1 ms GPU sum) — graph capture is the only viable decode mode.
  Entry: `docs/experience/wins/2026-08-24-sota-all-levers.md`.

## 2026-08-24 — default flip: paged_attention on sm90 is FlashAttention (was serial-scalar)

- **Default flip.** `paged_attention` in the sm90 cell is now
  `kernels_mma.make_paged_attention_mma` — the FlashAttention online-softmax
  schedule ported to paged KV + GQA, bf16 IO, block_M 16 (decode) / 64
  (prefill). The f32 serial-scalar kernel stays in kernels.py as the
  cpu/metal floor. Kernel-level at the 27B full-attn shapes (H=24, Hkv=4,
  D=256): decode M=1 KV=4096 37.84 → 0.456 ms (83x), prefill M=512 1056 →
  0.062 ms (17100x). Prefill is 35% of the bf16-tensor roofline under a
  99%-util co-tenant (within 2x idle); decode is still ~30x off the memory
  roofline — tilelang 0.1.13 lowers the paged gather to synchronous loads
  (ponytail: split-KV flash-decoding with pipelined gathers). Full-model
  impact: 16 full-attn layers add ~7.3 ms/tick decode (contended), ~0.002
  ms/tok prefill. Entry: `docs/experience/wins/2026-08-24-paged-attention-fa.md`.

## 2026-08-24 — default flip: fp8 prefill path on sm90 (e4m3 activations + fp4->e4m3 WGMMA)

- **Default flip.** `linear_fp4` with M>1 on sm90 now runs fp8 WGMMA (e4m3
  activations, e2m1fn→e4m3 weight dequant in the K-loop, f32 accumulate)
  instead of bf16 WGMMA — 1.5x on the slice prefill tick (6021 → 7839 tok/s,
  extrapolated 268 → 399 tok/s). pack_fp4's block scale moves per-16 → per-32
  to match the fp8 WGMMA K-tile (one scale per tile, no temp-fragment
  epilogue). e4m3's ~2% multiplicative quant error does not average down over
  K, so the parity gate uses an identical-quant torch reference. 399 tok/s is
  9.5x off the 3800 target — the kernel is dequant-bound (e4m3 cast in the
  K-loop, ~20% of fp8 peak); production fp8 GEMMs precompute fp8 weights.
  Entry: `docs/experience/wins/2026-08-24-fp8-prefill-wgmma.md`.

## 2026-08-24 — phase exit: decode tick captured (CUDA graph) on sm90

- **Exit + default flip.** The decode tick is now a captured kernel sequence
  (design-engine.md invariant): `_DecodeGraph` captures `model.forward` once
  per batch-size bucket (day-1 M=1) and replays per token — auto-on for CUDA,
  eager the default elsewhere and the fallback on capture failure. Dispatch
  drops from 899 ops x 20.4 us = 18.3 ms (full-model extrapolation) to
  0.040 ms (36 us pinned async copies + 3 us replay) on the 2-GDN-layer
  slice; the replay cost is op-count-independent. Two prerequisites landed
  with it: a `write_tokens` sm90 scatter kernel (the pool's host loop had
  per-token GPU->CPU syncs, uncapturable) and an on-device `_inv_freq` cache
  (the CPU-cached tensor H2D-copied on every rope, illegal in capture).
  Parity: eager vs captured token streams identical (tiny model, CUDA),
  `test_ops_parity.py` 26/26 on CUDA. Entry:
  `docs/experience/wins/2026-08-24-decode-graph-capture.md`.

## 2026-08-24 — verdict: bf16 IO + bitcast fast decode accepted on sm90

- **Verdict.** The sm90 MMA kernels (3 gemms + `linear_fp4_mma` +
  `linear_fp4_gemv`) switch from f32 to bf16 IO (bf16 WGMMA, f32 accumulate),
  and the e2m1fn decode switches from 4x `exp2` to integer bit-pattern
  synthesis (`sign<<31 | (126+e)<<23 | m<<22`, reinterpreted as float —
  ties the warp-shuffle LUT, no warp-cooperation constraint). Big-shape
  GEMV 1.8-2.1x faster (17408x5120: 0.138 -> 0.077 ms, 15% -> 27% of roof;
  lm_head 1.85 -> 0.89 ms, 33% of roof); WGMMA path 1.5x; slice decode
  4.55 -> 3.91 ms/tick. CUDA parity 25/25 at rtol=1e-2. Entry:
  `docs/experience/wins/2026-08-24-fp4-gemv-bitcast-bf16.md`.

## 2026-08-24 — verdict: multi-block norm/activation accepted on sm90

- **Verdict.** `silu_mul` gridded over M (1024-element chunks) and
  `rmsnorm` split-K (per-chunk partial sums + apply, two launches) in the
  portable floor — same source on CPU/CUDA/Metal, serial fragment-scalar
  accumulators (the example `T.reduce_sum` idiom is not Metal-portable).
  Slice prefill 512: silu_mul 46.1 -> 0.07 ms, tick 119.6 -> 73.7 ms
  (4257 -> 6886 tok/s). Decode rmsnorm 0.445 -> 0.410 ms — now
  launch-bound at 2 launches/call; next lever is fusion, not more blocks.
  CUDA parity 25/25. Entry:
  `docs/experience/wins/2026-08-24-multiblock-norm-act.md`.

## 2026-08-24 — phase exit: GEMV + chunk-kernel round closed on sm90

- **Exit.** Both kernels default-on, final slice numbers on H20: smoke
  8-token average 48.85 -> 31.09 -> 5.46 ms/tok, decode-only 5.335 ms/tick
  (187 tok/s), prefill-512 0.2226 ms/tok (4491 tok/s, 3800 slice target
  met). Full-model extrapolation (lm_head corrected): ~102 ms/tok decode
  (9.8 tok/s) and 6.95 ms/tok prefill (144 tok/s) vs 80/3800 targets —
  8.2x / 26x gap. Next levers per the profile: launch count (899 ops/tick,
  20 ms dispatch) and the single-block `silu_mul` grid (40% of prefill),
  not new GEMM schedules. Entry:
  `docs/experience/wins/2026-08-24-gemv-chunk-kernels.md`.

## 2026-08-24 — default flip: sm90 GDN prefill uses the fused chunk kernel

- **Flip.** `linear_attn_chunk` on sm90 now dispatches prefill (T>1) to
  `make_gdn_chunk_fused` (one launch per value head: conv1d + SiLU +
  q/k-norm + decay-first delta recurrence + gated RMSNorm + z-gate, serial
  scan over T) instead of the torch-eager reference (~150k tiny launches
  per 512-token prefill). T=1 keeps the decode kernel. 27B slice prefill:
  11.01 -> 0.2212 ms/tok (49.8x) on H20, JIT-free — the 3800 tok/s slice
  target is met. CUDA parity 25/25. Entry:
  `docs/experience/wins/2026-08-24-gdn-prefill-chunk.md`.

## 2026-08-24 — verdict: sm90 fp4 GEMV decode accepted (CUDA-verified)

- **Verdict.** `make_linear_fp4_gemv` (733cbcd, SOTA copy of
  `example_dequant_gemv_fp16xint4.py`) accepted as the sm90 M=1 decode
  path: CUDA parity 23/23, GEMV beats WGMMA-padded on every fp4 linear
  (1.3–5.9x), slice decode 10.577 -> 5.452 ms/tick (94.5 -> 183.4 tok/s)
  on H20. Roofline 12–14% on big shapes — headroom in decode ALU and
  launch count, not weight streaming. Entry:
  `docs/experience/wins/2026-08-24-fp4-gemv-decode.md`.

## 2026-08-24 — default flip: sm90 GDN decode uses the fused megakernel

- **Flip.** `linear_attn_chunk` on sm90 now dispatches decode (T=1) to
  `make_gdn_decode_fused` (one launch per value head: conv1d + SiLU +
  q/k-norm + decay-first delta recurrence + gated RMSNorm + z-gate, ported
  from `examples/gdn/qwen36_gdr_decode_fused.py` @ tilelang branch
  `feat/qwen36-gdn-megakernel`) instead of the torch-eager reference
  (~384 tiny launches/layer/tick). Prefill (T>1) keeps the reference.
  27B slice decode: 65.46 -> 47.16 ms/tok (28%) on H20, JIT-free.
  Entry: `docs/experience/wins/2026-08-24-gdn-decode-fused.md`.

## 2026-08-24 — default flip: sm90 cell switches from naive FMA to MMA (WGMMA)

- **Flip.** The sm90 cell now uses the MMA schedules in `kernels_mma.py`
  (shared-memory tiled `T.gemm` + pipelined K-loop, ported from
  `examples/gemm/example_gemm.py` and the Hopper dequant+gemm example) for
  `gemm_{nt,nn,tn}` and `linear_fp4`. The naive FMA schedules stay in
  `kernels.py` as the metal/other-arch fallback. 27B slice decode:
  1180.19 -> 48.85 ms/tok (24x) on H20, JIT-free. Entry:
  `docs/experience/wins/2026-08-24-sm90-mma-gemm.md`.
- **Format.** `pack_fp4`/`unpack_fp4` switched from the OCP e2m1 LUT (with
  zero) to e2m1fn (no zero) to match the kernel decode and the Hopper SOTA;
  `dequant_nvfp4` keeps the OCP grid (checkpoint wire format is separate).
  CUDA parity 21/21; CPU suite 64 passed, 1 skipped.

## 2026-08-24 — verdict: sm90 CUDA target accepted; real 27B slice forwards + trains

- **Verdict.** sm90 cell accepted: the 2-layer GDN slice of Qwen3.6-27B
  NVFP4 forwards through the engine (8 tokens, 1180.19 ms/tok JIT-free) and
  takes a training step (22.0 s/step, loss 11.2405 → 10.6960) on an H20,
  with CPU/CUDA logits parity to 6 decimals. 60 passed on CUDA. Entry:
  `docs/experience/wins/2026-08-24-sm90-real-slice.md`.
- **Fixes.** ModelOpt NVFP4 global scale is stored reciprocal (divide, not
  multiply); ModelOpt FP8-block `weight_scale_inv` is the scale itself
  (multiply, not divide) — both confirmed against agent-infer
  `quant_format.rs`, hermetic loader tests updated. `qwen36_27b()` is
  untied (checkpoint ships `lm_head.weight`). Bench warmup uses the timed
  `prompt_len` (JIT specializes per shape; a shorter warmup leaked NVCC
  into the measurement). sm90 registered with the naive FMA gemms;
  `Backend.device` pins `cuda:<current>`.
- **Bench (H20, tiny).** prefill 0.507 ms/tok (1973.5 tok/s), decode
  3.624 ms/tok (275.9 tok/s), prompt_len=128, gen=32, JIT-free.

## 2026-08-24 — format loaders: official NVFP4, per-tensor FP8, AWQ-int4; 23MB fixture retired

- **Features.** `load_hf` gains three formats (all dequant to bf16 at load):
  official NVIDIA NVFP4 naming (`weight` u8 nibbles + `weight_scale` f8 +
  scalar `weight_scale_2`; reuses the ModelOpt e2m1 math), per-tensor FP8
  (f8 `weight` + scalar `weight_scale` — the official-NVFP4 GDN/attn path and
  standalone FP8), and AWQ-int4 (`qweight`/`scales`/`qzeros`, autoawq GEMM
  packing, group size from `quantization_config`). `dequant_awq` added to
  `ops/reference.py`. Five formats now covered: bf16 HF, MLX-4bit, ModelOpt
  NVFP4/FP8-block, official NVFP4, FP8, AWQ-int4.
- **Tests.** 64 passed, 1 skipped on CPU (was 62+1). New synthetic
  per-format tests in `tests/test_weights.py` (KB-sized, formula-reference,
  `torch.equal`): `test_nvfp4_official_load`, `test_awq_load`,
  `test_mlx_affine_load`. Deleted: the 23MB `tests/fixtures/
  qwen35-2layer-mlx4/`, `tests/test_real_weights.py`, `scripts/crop_fixture.py`,
  and the orphaned `qwen35_08b` config. Entry:
  `docs/experience/wins/2026-08-24-format-loaders.md`.

## 2026-08-24 — pretrain loop + save_hf; ruff format gate turned on

- **Features.** `save_hf(model, path)` (model.py): HF safetensors + config.json
  roundtrip with `load_hf` — tensor names are the reverse of `_LAYER_SUFFIXES`,
  fp4 masters saved bf16 and re-packed on load. `pretrain(...)` +
  `JsonlDataset` (train.py): JSONL `"text"` corpus → eos-separated packed
  sequences, causal-LM loss via `train_step`, `cosine_warmup` wired in (its
  first production caller), seeded epoch-wise shuffle, periodic + final
  checkpoints. CLI: `tilerl pretrain --model tiny --data <jsonl> --steps N
  --seq-len 512 [--ckpt-dir D] [--ckpt-every M] [--lr] [--warmup] [--seed]`.
- **Default flip.** `ruff format --check` is a blocking CI gate
  (`continue-on-error` removed); tree reformatted (19 files), `ruff check
  --fix` clean.
- **Tests.** 60 passed, 3 skipped on CPU and Metal (was 58+3). New
  `tests/test_pretrain.py`: JSONL packing/padding, pretrain finite-loss +
  params-moved + checkpoint landing, save_hf → load_hf forward-match.

## 2026-08-24 — consolidation: -20% src LOC, decode exact, real weights coherent

- **LOC.** src 5671 → 4536 (-20.0%, hard target ≤4537 met); tests 1503 lines.
  Deletions: 7 selfcheck mains (-758), `weights.py` merged into `model.py`,
  `ops/fp4.py` merged into `ops/reference.py`, `testing.py` __getattr__
  delegation (-52), duplicated param-key mappers merged, docstring tightening.
- **Correctness fixes.**
  - **MLX 4-bit dequant formula** was `s*(q-b)`; the MLX quantized-matmul kernel
    uses `s*q+b` (scales may be negative). Real 0.8B next-token was gibberish,
    now exact (" Paris" for "The capital of France is").
  - **GDN conv1d carry** was zero-padded at every decode step (documented day-1
    limitation). `conv_window` [B,K-1,qkv_dim] now threads through
    `gdn_forward`/`LinearStatePool`/engine prefix snapshots; segmented decode is
    bit-exact vs one-shot prefill (parity test
    `test_gdn_conv_window_makes_step_exact`). Full 24-layer generation is
    coherent.
  - **Tape `add` handler was missing** — residual adds were unrecorded, so only
    embed/final_norm received grads (layers trained nothing). Caught by the new
    production-model gradcheck (AGENTS.md gate); fixed with a 2-line handler.
- **Features.** `load_hf(cfg, source, num_layers=N)` truncation (full-attn
  subset matched, skipped layers not required); precision×arch dispatch
  registry in `ops/backend.py` (`(precision, arch) -> kernel set`, fallback
  exact→any→any, GPU slots registered empty → NotImplementedError);
  `docs/support-matrix.md`.
- **Tests.** 55 passed, 3 skipped (GPU + 2 real-weight env-gated). New:
  conv-carry exactness, cosine_warmup, clip_grad_norm, production-model
  gradcheck, opd_loop smoke, white-box prefix-snapshot, engine miss-path,
  MLX dequant (via real-weight test), load_hf truncation.
- **Real weights (env-gated `TILERL_TEST_REAL=1`).** Qwen3.5-0.8B-MLX-4bit:
  truncated 2-layer forward + train_step pass; full 24-layer generation
  coherent ("The capital of France is Paris, and the capital of the United
  States is Washington, D.C.").
- **Bench (CPU, tiny).** prefill 45.68 ms/tok, decode 3.17 ms/tok
  (prompt_len=128, gen=32). Decode is far under the 60 ms/tok baseline — the
  conv-carry fix removed the per-step re-prefill penalty; prefill includes
  one-time JIT.
- **Pending-remote.** GPU arches (sm90/sm100/rocm/metal) registered as empty
  sets — NotImplementedError on use. 27B download test stays pending-remote.

## 2026-08-24 — phase exit: integration green, tiny model baseline

- **Full test suite green.** 45 passed, 1 skipped (test_gpu_targets — no CUDA
  on this host). Covers ops parity (tilelang vs torch-eager reference), KV
  pool lifecycle, e2e generation/prefix-cache/training/gradcheck/fp4, and the
  OpenAI-compatible server (health, models, non-stream, SSE stream).
- **Integration fixes.** Tape records structural ops (reshape/transpose/
  slice/add) as first-class entries so the id()-based grad chain never breaks;
  GDN layer is one monolithic backend op with a gradchecked torch-eager
  backward; dense training path (`kv.dense=True`) bypasses paged attention;
  fp4 pack/unpack moved to `ops/fp4.py`; engine prefix cache publishes all
  block-aligned prefixes; engine `_loop` guards against silent thread death.
- **Tiny model baseline (CPU).** prefill 56.98 ms/tok (17.6 tok/s), decode
  60.03 ms/tok (16.7 tok/s), prompt_len=128, gen=32. See
  `docs/experience/wins/2026-08-23-tiny-model-baseline.md`.
- **Smoke.** `tilerl --help`, `bench`, `train --steps 20`, `serve --model tiny`
  all run end-to-end. Server SSE streaming verified via curl.
- **Pending-remote.** GPU targets (cuda/rocm/metal) compile from the same
  kernel source but are unverified (no GPU on this host). 27B weights not
  downloaded (`TILERL_QWEN38_SOURCE` placeholder).

## 2026-08-23 — bootstrap: project scaffold and contract

- **TileLang-only backend.** One kernel source targeting cpu/cuda/rocm/metal;
  the numpy backend framing from the earlier partial bootstrap was deleted.
- **CPU target is the portable default and CI path.** This machine has no GPU;
  all verification runs on CPU. GPU targets are pending-host.
- **torch reduced to tensor container.** No `torch.autograd` / `torch.optim`;
  training runs on a hand-written reverse-mode autograd tape mirroring
  `agent-infer/crates/autograd`.
- **Engine seam**: submit/poll + `StepLimits`, continuous batching, one
  forward per tick. State: paged KV (full attention) + recurrent state
  (gated-delta) + hash prefix cache.
- **OPD training** shares the engine and weights with serving.
- **OpenAI-compatible server** entry point (`tilerl serve`).
- Docs scaffold: `AGENTS.md` (canonical agent contract; `CLAUDE.md` symlinks
  to it), `README.md`, bench-entry template under `docs/experience/wins/`.
