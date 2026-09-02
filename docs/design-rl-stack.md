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
  update path (`Adafactor.streams`) frees each `dW` inside backward. The
  weight itself must not be resident too: `W = U Σ₀ Vᵀ` is rebuilt per matrix
  and re-quantized, never held as a 50 GB bf16 master (today's full-parameter
  path holds exactly that master, `drop_quantized`). Budget: 35 (frames) + 15
  (served fp4) + one matrix in flight + activations ≈ 60–70 GB. Fits only
  under that streaming.
- ISO has no LoRA variant. LoRA + AdamW stays the day-1 adapter path; ISO is
  the full-parameter path. A low-rank frame rotation (`U⁺ = U₀ + A Bᵀ`
  retracted) is an experiment, not a commitment.

ISO-Merger is offline and checkpoint-only: SVD of `W₀` once (cached), frame
displacements per specialist, tangent projection, mask trailing 10% of modes,
one ridge-stabilized Gram solve, retract, reconstruct. LoRA specialists merge
too: `W_i = W₀ + B_i A_i` is a full matrix. It is `tilerl merge`.

Gates:
- CPU, tiny model: frame gradient vs a finite difference along a Stiefel
  tangent direction (the tape is hand-written; a wrong `G_U` is silent);
  ‖UᵀU − I‖ after retraction < 1e-4; singular values unchanged across steps;
  SFT loss falls.
- Pod, SFT first: steps to the same loss vs Adafactor on the 27B, peak < 96
  GB. ISO-RL only after per-step re-quantization exists (roadmap P3).
- `tilerl merge`, CPU: self-merge and Σ₀ preservation are true by
  construction (smoke); the gate is two tiny specialists each keeping their
  task better than plain averaging. Pod: two 27B specialists vs TIES and DARE
  (implemented here as baselines), MMLU flat.

Risks named: the paper stops at 8B and at bf16 bases; polar on 448 matrices per
step (7 × 64) is ~3 s at H20's 148 TFLOPS if done by SVD, which is why it is
Newton-Schulz; the fixed spectrum is the *quantized* base's spectrum, and
re-quantization steps off the fixed-spectrum family each step — the MMLU gate
is what catches that drift.

## 2. The draft head — DFlash 2, and keeping it on-policy

`incoai/Qwen3.8-27B-DFlash2` (mirror `z-lab/…`, Apache-2.0): a 2B block-
diffusion drafter, 8-token blocks (7 drafts per verify), acceptance length 5.46
on GSM8K, **3.43× at concurrency 1 and 2.84× at 8 — sglang's numbers with
sglang's integration**, not a property of the head. The checkpoint also ships
its own MTP head. `spec.py`'s `DraftHead` is a one-token-per-tick chain with
its own KV pool; a block drafter that emits 7 tokens in one forward is a
second draft seam beside it, not a drop-in. `verify_lens`'s per-row trimming
does not survive graph capture either — a captured tick pads to one width.

Nobody needs tileRL to *train* a head — SpecForge and the DFlash release do
that. What nobody does: **the head trained on the current policy's rollouts,
on the same tape, so it does not go stale as the adapter moves.** Rollouts are
the RL samples we already have; the target hidden states the head conditions on
come out of the training forward for free; the head's block-diffusion loss is
one more `grad_fn`. A 2B head with Adafactor is ~10 GB alongside the 27B.

Order, forced by the measurement above:

0. **Recapture after each update.** The RL rollout runs eager today because a
   captured graph bakes weights the optimizer moves; re-record the graph after
   every step and the rollout runs captured. Every later baseline is this.
1. **Capture the speculative tick.** Until draft + verify replay from a graph,
   every head loses 4.9× to plain decode. Use the checkpoint's own MTP head —
   nothing to train. Exit: seconds per RL step at group 8 ≥ 1.5× better than
   captured plain decode (not eager).
2. DFlash2 through its own block-draft seam; same table; ≥ 1.2× over the MTP
   head or it is not kept.
3. Co-train it. Needs the head's bf16 master (the engine quantizes the draft
   to fp8 and drops the params) and `hidden_out` through `train._step`. Exit:
   acceptance length at the end of a 100-step run ≥ 0.9× its step-0 value.

Physics that bounds the payoff: speculation and batch are substitutes. 3.43×
at c=1 becomes 2.84× at c=8 and keeps falling as the verify batch turns
compute-bound (H20: 148 TFLOPS). B ≤ 8 rollouts are where the head pays; the
B ≥ 32 decode GEMM (roadmap P6) is the other regime's lever. Pick per regime,
do not stack.

## 3. The ledger CLI — the agent's surface

Every command writes `runs/<id>/manifest.json`: base hash, data hash,
algorithm, hyperparameters, gates with pass/fail, metrics, artifact paths,
parent run ids. `id = hash(inputs)` where inputs include the checkpoint path, the commit and
the seed (checkpoint bytes are not hashed — 15 GB — path + commit is the
honest cheap identity), so a rerun is a no-op and a changed input is a new
run. A failed gate is a non-zero exit. `--json` on everything.

```
tilerl train   --rl|--opd --data <jsonl> [--optim iso] [--draft ...] [--eval-mmlu N --eval-gsm8k <jsonl>]
tilerl merge   --base <dir> --specialists <dir>,<dir> --method iso|average
tilerl serve   --run <id>                       # base + adapter (+ head)
tilerl ledger  [--lineage <id>] [--json]        # what exists, what it descends from
```

Command names stay as they are in the tree (`train`, `merge`, `serve`,
`ledger`); `eval` is a flag on `train` until a run needs re-scoring alone.

`docs/experience/` stays the human record; the manifest is the machine one.
The static page is one HTML rendered from the manifests, built after a second
human user exists, never before.

## What this does not change

The verdict rule in `roadmap.md` P5 stands: the same-pod comparison vs
verl+sglang decides whether the engine is the product. It runs after the
rollout is captured (P2.0) and prices **seconds to the P1 target**, not
seconds per step — the honest unit once steps are not equal. Arm B runs the
FP8 checkpoint because sglang cannot load NVFP4 on Hopper: same task, not the
same bytes, and the table says so.
