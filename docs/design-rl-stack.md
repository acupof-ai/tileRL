# The RL stack — ISO optimizer + merger, an on-policy draft head, a ledger CLI

Status 2026-09-02: direction accepted, nothing below is built. This doc says
what each piece is, how it lands on the tape and the engine we already have,
what physics constrains it, and the gate each one exits on. Order is by
dependency; `docs/roadmap.md` carries the phases.

The product, in one line: **one card, one process, one command — train a draft
head, RL, evaluate, merge, serve.** The agent drives the CLI; a human reads a
page rendered from the ledger. The front end is last and static.

## What is settled and what is not

| | settled | open |
|---|---|---|
| RL loop | GRPO through the engine, LoRA on the frozen fp4 base, real task + reward + MMLU gate wired (`wins/2026-09-02-rl-real-task.md`) | 27B run is pending-remote; the loop has never moved a downstream metric |
| Speculation | draft + verify in one trunk forward, 1.87 committed tokens/forward, 43–47% acceptance on the tiny probe (`CHANGELOG 2026-08-29`) | **a draft disables graph capture: 86.2 tok/s captured vs 17.6 speculating.** Spec loses 4.9× on this engine until the spec tick is captured |
| Optimizer | AdamW (adapters), Adafactor + streamed updates (full-param fits 73 GB) | ISO: not on the tape, untested above 8B, untested on a quantized base |
| Merging | none | ISO-Merger vs TIES/DARE on our own specialists |
| Ledger | `docs/experience/` + `bench-baseline.json`, human-written | no per-run manifest, no lineage, no machine-readable gates |

## 1. ISO — the optimizer and the merger

Paper: *ISO: An RLVR-Native Optimization Stack* (arXiv 2607.19331, Jul 2026;
code `github.com/zhuhanqing/ISO`). Finding: RLVR keeps the base model's
singular spectrum and changes only the singular frames. So each linear is
parametrized `W = U Σ₀ Vᵀ` with `Σ₀` frozen from the base and `U, V` on the
Stiefel manifold; the frame gradients are `G_U = G_W V Σ₀`, `G_V = G_Wᵀ U Σ₀`;
any base optimizer (AdamW, Muon) steps `U, V`; a polar retraction restores
orthonormality. Reported: Qwen3-4B/8B reach AdamW's endpoint in **~2.7× fewer
steps**; the retraction is ~7% of an RLVR step. ISO-Merger composes the frame
displacements of specialists sharing a base — no data, no rollouts — and beats
the best data-free baseline by 1.6 pt on Qwen2.5-7B (3 experts) and 0.9 pt on
R1-Distill-1.5B.

Why it matters here more than in the paper: in tileRL's cost model RL time is
rollout time (`plan-training-rl.md` §2). 2.7× fewer steps is 2.7× less rollout,
before any kernel work.

How it lands on the tape — optimizer-side only, no new backward:

- The linear's `dW` is already produced. An `ISO` optimizer wrapper turns it
  into `(G_U, G_V)`, steps `U, V` with the base optimizer, retracts, rebuilds
  `W = U Σ₀ Vᵀ`, and hands the served copy to the existing re-quantize path
  (`plan-training-rl.md` P5: twiddled fp4 per step, **MMLU-gated** because
  quantization noise does not show in loss).
- Retraction is Newton-Schulz polar (matmuls only, what Muon uses), not SVD:
  graph-capturable and a TileLang kernel when it matters.
  `# ponytail: torch-eager Newton-Schulz, tilelang kernel when perf demands`
- **Memory is the constraint, and it forces Adafactor.** `U, V` for the 27B
  are ~1.3–1.5× the weight bytes (gate_up 34816×5120 → 1.15×, down → 1.29×,
  square projections → 2×): ~35 GB bf16. AdamW states on top are ~140 GB —
  does not fit a 96 GB card. Adafactor on `U, V` is ~0 GB, and the streamed
  update path (`Adafactor.streams`) frees each `dW` inside backward. Budget:
  35 (frames) + 15 (served fp4) + activations ≈ 60–70 GB. Fits.
- ISO has no LoRA variant. LoRA + AdamW stays the day-1 adapter path; ISO is
  the full-parameter path. A low-rank frame rotation (`U⁺ = U₀ + A Bᵀ`
  retracted) is an experiment, not a commitment.

ISO-Merger is offline and checkpoint-only: SVD of `W₀` once (cached), frame
displacements per specialist, tangent projection, mask trailing 10% of modes,
one ridge-stabilized Gram solve, retract, reconstruct. LoRA specialists merge
too: `W_i = W₀ + B_i A_i` is a full matrix. It is `tilerl merge`.

Gates:
- frame gradient handler: numerical gradcheck on the tiny model (the tape is
  hand-written; a wrong `G_U` is silent).
- ISO-Adafactor vs LoRA-AdamW on GSM8K, same rollouts: steps to the same
  reward; MMLU flat after re-quantization; peak memory < 96 GB.
- `tilerl merge` on two of our own specialists vs TIES and DARE (the paper's
  baselines): each specialist's task retained, MMLU flat.

Risks named: the paper stops at 8B and at bf16 bases; polar on 448 matrices per
step (7 × 64) is ~3 s at H20's 148 TFLOPS if done by SVD, which is why it is
Newton-Schulz; the fixed spectrum is the *quantized* base's spectrum, and
re-quantization steps off the fixed-spectrum family each step — the MMLU gate
is what catches that drift.

## 2. The draft head — DFlash 2, and keeping it on-policy

`incoai/Qwen3.8-27B-DFlash2` (mirror `z-lab/…`, Apache-2.0): a 2B block-
diffusion drafter, 8-token blocks (7 drafts per verify), acceptance length 5.46
on GSM8K, **3.43× at concurrency 1 and 2.84× at 8** on sglang. The checkpoint
also ships its own MTP head. `spec.py` already anticipates a DFlash-style head
(no confidence head; the draft's own softmax is the survival fallback).

Nobody needs tileRL to *train* a head — SpecForge and the DFlash release do
that. What nobody does: **the head trained on the current policy's rollouts,
on the same tape, so it does not go stale as the adapter moves.** Rollouts are
the RL samples we already have; the target hidden states the head conditions on
come out of the training forward for free; the head's block-diffusion loss is
one more `grad_fn`. A 2B head with Adafactor is ~10 GB alongside the 27B.

Order, forced by the measurement above:

1. **Capture the speculative tick.** Until draft + verify replay from a graph,
   every head loses 4.9× to plain decode. Use the checkpoint's own MTP head —
   nothing to train. Exit: seconds per RL step ≥ 1.5× better than plain decode
   at group 8.
2. Vendor DFlash2 behind the same `DraftHead` seam; same table, same gate.
3. Co-train it. Exit: acceptance length after N RL steps with co-training ≥
   the frozen head's acceptance at step 0, where the frozen head has decayed.

Physics that bounds the payoff: speculation and batch are substitutes. 3.43×
at c=1 becomes 2.84× at c=8 and keeps falling as the verify batch turns
compute-bound (H20: 148 TFLOPS). B ≤ 8 rollouts are where the head pays; the
B ≥ 32 decode GEMM (roadmap P6) is the other regime's lever. Pick per regime,
do not stack.

## 3. The ledger CLI — the agent's surface

Every command writes `runs/<id>/manifest.json`: base hash, data hash,
algorithm, hyperparameters, gates with pass/fail, metrics, artifact paths,
parent run ids. `id = hash(inputs)`, so a rerun is a no-op and a changed input
is a new run. A failed gate is a non-zero exit. `--json` on everything.

```
tilerl rl      --base <ckpt> --data <jsonl> --algo grpo|iso [--adapter lora|full]
tilerl head    --base <ckpt> --draft mtp|dflash2 [--cotrain]
tilerl eval    --run <id> --suite mmlu,gsm8k
tilerl merge   --runs <id>,<id> --method iso|ties|dare --select-by <suite>
tilerl serve   --run <id>                       # base + adapter (+ head)
tilerl ledger  [--lineage <id>]                 # what exists, what it descends from
```

`docs/experience/` stays the human record; the manifest is the machine one.
The static page is one HTML rendered from the manifests, built after a second
human user exists, never before.

## What this does not change

The verdict rule in `roadmap.md` P2 stands: the same-pod comparison vs
verl+sglang decides whether the engine is the product. With ISO and a captured
speculative tick the number to beat becomes **seconds to a target reward**,
not seconds per step — that is the honest unit once steps are not equal.
