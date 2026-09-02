# Roadmap

**North star.** Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card in
one process. What exists: the engine that samples is the model that trains,
LoRA on the frozen fp4 base, so there is no weight sync between rollout and
update. What is designed and not built: fixed-spectrum full-parameter updates
(ISO) re-quantized into the served fp4 bytes each step. The product is one CLI
an agent drives — `train → eval → merge → serve` — with every run recorded in
a ledger; a human reads a page rendered from that ledger.

Phases exit on a named, measurable event, never on a date. A gate that needs
a GPU not in hand ships its code with `pending-remote` in the wins entry and
does not claim the number. Detail: `docs/design-rl-stack.md` (ISO, the draft
head, the ledger); TP, CP and the 128K–256K budget are under P6 below.

## Where we are (2026-09-02)

| Area | State | Evidence |
|---|---|---|
| Serving | H20 B=1 decode 92.4 tok/s (sglang bf16 54.2, Arle 84.5); B=8 0.8× sglang; prefill 0.4× | `wins/2026-08-28-decode-split-by-occupancy.md`, `wins/bench-baseline.json`, `docs/experience/2026-08-28-vs-sglang-h20.md` |
| Accuracy | MMLU 0-shot 76.3% (1000 q) | `wins/2026-08-28-mmlu-letter-restricted.md` |
| Speculation | correct, 1.87 committed tokens per trunk forward; **loses 4.9× because a draft disables graph capture** | CHANGELOG 2026-08-29 verdict |
| Training | LoRA-AdamW and Adafactor full fine-tune run on one card (73.2 GiB); GRPO and self-OPD exist; real prompts, GSM8K reward, MMLU before/after wired | `wins/2026-08-29-full-finetune-fits.md`, `wins/2026-09-02-rl-real-task.md` |
| RL on the 27B | **never moved a downstream metric**; the run is pending-remote (pod held by another job) | same |
| Kernels | one TileLang tree; cpu (CI/parity), metal, sm90 executed it; 71% of kernel lines are sm90 schedules | `docs/support-matrix.md` |
| Ledger | human-written `docs/experience/`; per-run manifests landing 2026-09-02 (P4) | — |

## P1 — RL moves a number on the 27B — needs the pod

Everything below is built on this claim, and it is unproven.

Prerequisites, all code, all CPU-gated:
- `--eval-gsm8k` on a held-out slice (P4 builds it): today only MMLU exists.
- The OPD teacher engine is built with the decode graph and the prefix store
  ON, and the EMA adapter is swapped by `params.update`, which replaces the
  tensor objects a captured graph holds — on CUDA self-OPD samples from a
  stale policy without raising. Same class of bug GRPO already fixed
  (`wins/2026-08-29-grpo.md`, "Two ways"). Fix: `decode_graph=False`,
  `NoPrefixStore()`, `copy_` into the adapter tensors.
- Log the fraction of tied groups (all-equal rewards give zero advantage,
  `train.py group_advantages`); a no-think 27B on GSM8K may tie most groups.

Run: `tilerl train --recipe grpo-gsm8k-27b --data gsm8k_train.jsonl
--eval-gsm8k gsm8k_test.jsonl` (the recipe is 100 steps, group 8, 256 tokens,
LoRA rank 16, no thinking, MMLU 1000, GSM8K 500), one H20, `--seed 0` and `1`.

Exit, both seeds: GSM8K held-out (500 q) after − before ≥ +5 pt (SE ≈ 2 pt);
MMLU (1000 q) after ≥ before − 2 pt; tied-group fraction < 50% (else the task
is too easy for this model and the run says nothing — move to MATH). Then
self-OPD, same gate. The manifest (P4) records the verdict.

## P2 — the speculative tick is captured, and the head stays on-policy — needs the pod

Rollout is the RL cost, rollout is decode, speculation is the decode lever at
B ≤ 8. Today any draft head loses 4.9× (86.2 → 17.6 tok/s) because the
speculative tick runs eager. So does the RL rollout itself: `grpo_loop`
requires `decode_graph=False` because a captured graph bakes weights the
optimizer moves. Both need the same thing first:

0. **Recapture after each update.** The captured decode graph, and later the
   captured speculative tick, are re-recorded after every optimizer step
   (engine.py's "graph bakes the f32 cast" ponytail). Then RL rollouts run
   captured. Exit: rollout tokens/s at group 8 equals plain captured decode
   within 5%, and the sampled distribution matches eager (same seeds, same
   tokens on the tiny model).
1. Capture draft + verify with the checkpoint's own MTP head. Baseline is the
   captured plain decode of step 0, not eager. Exit: seconds per RL step at
   group 8 improves ≥ 1.5× over that baseline; acceptance length measured
   here on GSM8K ≥ 3.
2. DFlash2 (`incoai/Qwen3.8-27B-DFlash2`, 2B, Apache-2.0): a block drafter
   that emits 7 tokens in one forward. That is not the `DraftHead` chain (one
   token per tick, own KV pool), so it is a second draft seam, not a drop-in.
   sglang reports 3.43× at c=1 with its own integration; ours is measured
   here. Exit: same table as step 1; ≥ 1.2× over the MTP head or it is not
   kept.
3. Co-train the head on the RL rollouts on the same tape. Needs the head's
   bf16 master (the engine quantizes the draft to fp8 and drops the params)
   and the trunk's `hidden_out` through `train._step`. Exit: acceptance length
   at the end of a 100-step run ≥ 0.9× its value at step 0 (absolute, so an
   un-decayed frozen head cannot make the gate vacuous).

## P3 — ISO on the tape: optimizer, then merger — CPU today, 27B on the pod

ISO (arXiv 2607.19331) keeps the base spectrum and moves the singular frames;
optimizer-side only, no new backward. Reported 2.7× fewer steps at 4B/8B in
bf16 — a reason to try it, not a number we own. Mechanism and memory in
`design-rl-stack.md` §1.

- Optimizer (CPU, today): frame gradients from `dW`, Newton-Schulz polar,
  Adafactor base, streamed updates. Exit: tiny-model gradcheck of the frame
  gradient, orthonormality after retraction, spectrum preserved over steps,
  SFT loss falls.
- Optimizer (pod, SFT first): steps to the same loss vs Adafactor on the 27B;
  peak < 96 GB. This is SFT because full-parameter RL has a ceiling:
- **Per-step re-quantization into the served fp4 bytes** — the first pod item
  of this phase, before any ISO-RL number. Today full-parameter training
  drops the quantized bytes and runs the tape on bf16 masters
  (`drop_quantized`), so a rollout after a full-parameter step is off-policy.
  Budget must include the weight itself: frames ~35 GB + served fp4 15 GB +
  `W = U Σ₀ Vᵀ` rebuilt per matrix and re-quantized, never resident as a
  50 GB master. Exit: rollout logits after a step match the re-quantized
  weights; MMLU flat; peak < 96 GB. Then ISO-RL: steps to the P1 target vs
  LoRA-AdamW.
- Merger (CPU, today): `tilerl merge --method iso`. Self-merge and Σ₀
  preservation are true by construction and are smoke checks; the gate is two
  tiny specialists each keeping their task better than plain averaging.
- Merger (pod): two 27B specialists need a second task with its own reward
  and eval (MATH or code — to be chosen at P1 exit); beat TIES and DARE,
  which we implement as baselines; MMLU flat.

## P4 — the ledger CLI — CPU, today

The agent's surface. `runs/<id>/manifest.json` per run: inputs (including
checkpoint path, commit, seed), data hash, parents, metrics, gates with
pass/fail, artifacts. `id = hash(inputs)`, so a rerun is a no-op; a failed
gate is a non-zero exit; `--json` everywhere. Command names stay `train`,
`merge`, `eval`, `serve`, `ledger`; there is no rename.

- Exit: `tilerl train` and `tilerl merge` write manifests; `tilerl ledger`
  lists runs and lineage; an identical rerun does nothing; a test chains
  train → ledger on the tiny model. Lands before the P1 run so P1's verdict is
  machine-read.
- The static page rendered from manifests comes after a second human user
  exists, never before.

## P5 — same pod, same task, vs verl + sglang — needs the pod, after P2.0

Recurring (1 d) once P1 lands; OpenRLHF and slime are the next arms after verl.

- `scripts/rl_compare.sh`: today it records median seconds per step and
  MMLU before/after; the target-accuracy timing needs P1's target. The verl
  arm is an unverified command, and sglang cannot load the NVFP4 checkpoint on
  Hopper — arm B runs the FP8 checkpoint (27 GB; the bf16 one is 54 GB and
  emits garbage). Same task, each engine's best supported format on this card,
  **not the same bytes** — the table says so.
- Runs after P2.0 so it judges a captured rollout, not the eager one.
- Verdict rule, decided now: if tileRL is not faster to the P1 target under
  those conditions, the engine stops being the product and `tilerl-kernels`
  (w4a8 NVFP4 on Hopper) goes upstream as a quantization backend PR.

## P6 — rollout decode at B ≥ 32, then TP-8, then CP — needs the pod

Speculation and batch are substitutes: 3.43× at c=1 is 2.84× at c=8 and
falls as the verify batch turns compute-bound. B ≥ 32 only happens when a
step samples ≥ 32 rollouts — `grpo_loop` is one prompt per step today, so this
lever needs multi-prompt steps first. Choose P2 or P6 after P1 by the group
size the task needs.

- Tensor-core decode GEMM MX 8 → 32/64. Exit: harness B=32 row at the
  rollout's shape (≤ 512-token context) ≥ 3× the B=8 aggregate.
- TP-8, one KV head per card. Exit: TP-8 B=1 tick ≈ 3 ms; loss bit-identical
  to TP-1 on tiny; IPC/NCCL crossover measured.
- CP-8: ring attention for full attention, prefix scan for GDN, head/tail
  balanced. Exit: 256K fwd+bwd on 8 cards; 32K gradients match a single
  card to 1e-3; the scan gradchecked on tiny first.
- Full-parameter on the pod: ZeRO-1 masters, re-quantize to twiddled fp4
  each step. Exit: ≤ 60 GB/card at 32K; MFU ≥ 40%; MMLU flat.

Physics that fixes the design:

- Decode is bandwidth-bound (22.8 GB per tick at ~64% of 3.25 TB/s);
  training is compute-bound (H20 ~148 TFLOPS bf16, 6·N ≈ 126 GFLOP/token ⇒
  ~590 tok/s/card at 50% MFU, ~4.7k tok/s on 8 cards, a 32K sample ≈ 7 s).
  fp4 weights buy nothing for training; the levers are backward kernels and
  MFU. A 32K rollout at B=1 is 6 min against a 7 s update — RL time is decode.
- TP degree is chosen by KV memory: per token the 16 full-attention layers
  hold 16 × 8 heads × 256 × 2 B × (K+V) = 131 KB (the 48 GDN layers hold a
  constant state, which is why 256K is feasible at all). 8 KV heads, 8 cards
  ⇒ TP-8 with one KV head per card; it also shards the GDN value heads (48/8)
  and the weights (22.8/8 GB).
- Decode TP is 10 KB per all-reduce, 128 times per tick; NCCL's ~15 µs floor
  (21.5 µs measured, CHANGELOG 2026-08-30) is ~2 ms of a ~3 ms TP-8 tick. A
  one-shot CUDA-IPC all-reduce as a TileLang kernel is ~3–5 µs and
  graph-capturable; training traffic (grad all-reduce, ZeRO, CP ring) stays
  on NCCL. One `comm.py` seam, crossover measured by microbench, IPC falls
  back to NCCL when peers are not mappable.
- CP for GDN is a scan, not a hand-off: `S_i = A_i S_{i-1} + B_i` composes,
  so each rank computes local (A, B), one all-gather fixes the incoming
  state, a second pass produces outputs; backward is the same scan reversed.
  Full-attention CP is ring attention and its merge is the split-KV combine
  already shipped. From torchtitan: mesh-first `ParallelDims`, head/tail
  causal balancing, seed-by-name init, fwd+bwd in one CUDA graph; DTensor and
  `fully_shard` are torch.autograd machinery and do not transfer.
- Risks: NCCL inside captured decode graphs needs stream discipline (fallback
  a graph per layer); re-quantization noise per step is invisible in loss,
  MMLU-gated; the GDN scan is the one novel kernel and does not touch the 27B
  until the tiny gradcheck passes.

128K–256K budget:

| | 32K | 128K | 256K |
|---|---:|---:|---:|
| KV per sequence | 4.3 GB | 17 GB | 34 GB |
| one card (22.8 GB weights): concurrent rollouts | 16 | 4 | **1** (57 GB) |
| TP-8: concurrent rollouts per pod | 128 | 32 | 16 |
| decode KV read per token / card, TP-1 → TP-8 | 1.3 → 0.2 ms | 5.3 → 0.7 ms | 10.6 → 1.3 ms |
| B=1 decode (11 ms weights tick + KV at ~65%): TP-1 / TP-8 | 77 (measured) / — | ~50 / ~80 tok/s | ~40 / ~80 tok/s |
| prefill attention FLOPs 2·n²·d·H·L | 0.3 PF | 4.5 PF | 18 PF ⇒ ~3 min one card at 100 TF, ~25 s TP-8 |
| training a 256K sample, CP-8 + per-layer activation checkpoint | | | ~20 GB/card activations, ~2 min fwd+bwd per pod |

What the budget forces: single-card 256K first (`max_total_tokens` 262144,
16 384 blocks per sequence, split-KV beyond 16 splits at depth, harness rows
d131072/d262144 at B=1; gate: the 256K→512-token decode curve is explained by
KV bytes alone); the prefix cache must hit across a rollout batch at 128K
(gate: kv-reuse hit rate 1.0, warm ≪ cold); prefill attention at 256K is 18
PFLOP, so the dense MMA prefill needs a FlashAttention-3-class schedule or
TP-8 hides it — today's 1.8k tok/s at 8K says nothing about 256K; CP-8 +
activation checkpointing is the only way a 256K sample trains (without
recompute the activations are ~1.6 TB).

## Dependencies

```
P4 ledger (CPU, now) ─┐
P3 CPU half (now)     ├─► P1 (RL moves a number) ─┬─► P2.0 recapture ─► P2.1–3 heads ─► P5 verdict
                      │                           ├─► P3 pod: re-quant ─► ISO-RL, merger
                      └───────────────────────────┴─► P6 (B≥32 needs multi-prompt steps; TP; CP)
```

## Parked (trigger-based)

- **Serve throughput at B ≥ 8 and prefill vs sglang** — sglang's home turf,
  not what an RL runtime is priced on
  (`docs/experience/2026-08-29-what-sota-would-require.md`).
- **PD / AFD** — design only (`docs/design-pd-afd.md`); trigger: one engine
  is batch-limited.
- **GDN2** — trigger: a GDN2 checkpoint in the target family
  (`docs/design-gdn2.md`).
- **Other SMs and ROCm** — sm100/sm120 are registered and empty; sm86/89 are
  unregistered; ROCm has no cell. Trigger: hardware in hand. The claim to make
  then is "a new arch is a bounded job" (register a cell, run the suite, tune
  the compute-bound kernels), not "runs everywhere".
- **Web front end** — trigger: a second human user.
