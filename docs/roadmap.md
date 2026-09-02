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
head, the ledger), `docs/plan-training-rl.md` (TP, CP, 128K–256K budgets).

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

Run: `tilerl train --model qwen38-27b --rl --data gsm8k_train.jsonl --steps
100 --group 8 --max-new-tokens 256 --eval-mmlu 1000 --eval-gsm8k
gsm8k_test.jsonl --eval-n 500`, LoRA rank 16, no thinking, one H20, seeds 0
and 1.

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
- TP-8 (one KV head per card, capturable all-reduce — CHANGELOG 2026-08-30),
  then CP (ring attention + GDN prefix scan) for 128K–256K. Exits in
  `docs/plan-training-rl.md`.

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
