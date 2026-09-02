# Roadmap

North star: serve and RL-train Qwen3.x-27B (NVFP4) on one Hopper node in one
process — the engine that samples is the model that trains, LoRA on the frozen
fp4 base, so there is no weight sync between rollout and update. Phases are
ordered by the risk each retires; each exits on a named, verifiable event.

## Where we are (2026-09-02)

| Area | State |
|---|---|
| Serving | H20 B=1 decode 92.4 tok/s (sglang bf16 54.2, Arle 84.5); B=8 0.8× sglang; prefill 0.4×; MMLU 76.3% |
| Training | LoRA and Adafactor full fine-tune run on one card; GRPO and self-OPD exist; `--data` real prompts + GSM8K reward + `--eval-mmlu` gate shipped **pending-remote** |
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
  Then self-OPD the same way, same gate.

## P2 — same pod, same task, vs verl + sglang

- `scripts/rl_compare.sh`: seconds per RL step (rollout + update) at group 8 ×
  256 tokens, then at 32K context; MMLU before/after on both arms.
- Exit: the table exists, recorded like the serving comparison
  (`docs/experience/2026-08-28-vs-sglang-h20.md`).
- Verdict rule, decided now: if tileRL is not faster per step, the engine
  stops being the product and `tilerl-kernels` (w4a8 NVFP4 on Hopper) goes
  upstream as a quantization backend PR.

## P3 — rollout decode at B≥32

RL time is rollout time, and rollout is a decode kernel at B≥32, not B=1.

- Tensor-core decode GEMM from MX=8 to 32/64; recapture the decode graph and
  drop prefix entries after each update instead of eager rollouts.
- Exit: harness B=32 row ≥ 3× the B=8 aggregate; RL step time falls by the
  rollout's share.

## P4 — TP-8, then CP for 128K–256K

`docs/plan-training-rl.md` P3–P4: one KV head per card, a capturable
all-reduce, ring attention + the GDN prefix scan. Exit conditions there.

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
