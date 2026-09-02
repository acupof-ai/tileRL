# Roadmap

North star: serve and RL-train Qwen3.x-27B (NVFP4) on one Hopper node in one
process — the engine that samples is the model that trains, LoRA on the frozen
fp4 base, so there is no weight sync between rollout and update. Phases are
ordered by the risk each retires; each exits on a named, verifiable event.

## Where we are (2026-09-02)

| Area | State |
|---|---|
| Serving | H20 B=1 decode 92.4 tok/s (sglang bf16 54.2, Arle 84.5); B=8 0.8× sglang; prefill 0.4×; MMLU 76.3% |
| Training | LoRA and Adafactor full fine-tune run on one card; GRPO and self-OPD exist; `--data` real prompts + GSM8K reward + `--eval-mmlu` gate shipped **pending-remote**; ISO and the on-policy draft head are designed (`docs/design-rl-stack.md`), not built |
| Kernels | one TileLang tree; cpu (CI/parity), metal, sm90 executed it; 71% of kernel lines are sm90 schedules |
| Tests | hermetic CPU suite on ubuntu + macos, every commit |
| Adoption | 1 star, not on PyPI |

## P1 — training is real on the 27B

The differentiating claim is unproven until a run moves a number.

- `tilerl train --model qwen38-27b --rl --data gsm8k.jsonl --group 8
  --max-new-tokens 256 --eval-mmlu 200` (`scripts/gsm8k_jsonl.py` dumps the
  data). LoRA rank 16, no thinking, one H20.
- Exit: GSM8K reward rises from its step-0 value and holds; MMLU after ≥ MMLU
  before within noise (200 questions ≈ ±3 pt); the curve in a wins entry.

## P2 — the speculative tick is captured, and the head stays on-policy

`docs/design-rl-stack.md` §2. A draft disables graph capture today, so every
head loses 4.9× to plain decode.

- Capture draft + verify with the checkpoint's own MTP head. Exit: seconds per
  RL step at group 8 ≥ 1.5× better than plain decode.
- Vendor DFlash2 (`incoai/Qwen3.8-27B-DFlash2`, 3.43× at c=1 on sglang)
  behind the same `DraftHead` seam; same table.
- Co-train the head on the RL rollouts. Exit: acceptance length does not decay
  over the run where the frozen head's does.

## P3 — ISO on the tape: optimizer, then merger

`docs/design-rl-stack.md` §1. Optimizer-side only; no new backward.

- Frame gradients + Newton-Schulz polar + Adafactor + streamed updates +
  per-step re-quantization. Exit: tiny-model gradcheck of the frame handler;
  steps to the same GSM8K reward vs LoRA-AdamW (paper: 2.7× fewer); MMLU flat;
  peak < 96 GB.
- `tilerl merge --method iso` on two of our own specialists. Exit: beats TIES
  and DARE on the same pair, MMLU flat.

## P4 — the ledger CLI

`docs/design-rl-stack.md` §3. Per-run manifest, gates as exit codes, `--json`,
lineage. Cheap, no GPU, can run alongside P1–P3. Exit: an agent runs
rl → eval → merge → serve from manifests alone.

## P5 — same pod, same task, vs verl + sglang

- `scripts/rl_compare.sh`: seconds to a target GSM8K reward (steps are not
  equal once ISO and speculation are in), MMLU before/after, both arms.
- Verdict rule, decided now: if tileRL is not faster to the target, the engine
  stops being the product and `tilerl-kernels` (w4a8 NVFP4 on Hopper) goes
  upstream as a quantization backend PR.

## P6 — rollout decode at B≥32, then TP-8 / CP

Speculation and batch are substitutes; this is the other regime's lever.
Tensor-core decode GEMM MX 8 → 32/64 (exit: harness B=32 row ≥ 3× the B=8
aggregate), then `docs/plan-training-rl.md` P3–P4 for 128K–256K.

## Parked (trigger-based)

- **Serve throughput at B≥8 and prefill vs sglang** — parked. That is
  sglang's home turf and not what an RL runtime is priced on
  (`docs/experience/2026-08-29-what-sota-would-require.md`).
- **PD / AFD** — design only (`docs/design-pd-afd.md`); trigger: a single
  engine is batch-limited.
- **GDN2** — trigger: a GDN2 checkpoint in the target family
  (`docs/design-gdn2.md`).
- **fp8 / sm100 cells, ROCm** — trigger: hardware in hand. ROCm has no
  registry cell until a HIP host runs the suite.

## Standing discipline

- Every hot-path change ships a bench entry (`docs/experience/wins/` or
  `errors/`).
- New op ⇒ parity check; new backward ⇒ numerical gradcheck. No exceptions.
- LOC is the thing to cut; a shorter diff passing the same gates wins.
