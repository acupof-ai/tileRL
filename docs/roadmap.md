# Roadmap

North star: one TileLang kernel source trains and serves Qwen3.8-27B (NVFP4)
on cpu / cuda / rocm / metal. Phases are ordered by risk retired per unit
work; each exits on a named, verifiable event — no calendar dates.

## Where we are (2026-08-24)

| Area | State |
|---|---|
| Backend | TileLang single backend; CPU target green in CI (ubuntu + macos), Metal green locally |
| Model | Hybrid full-attn + GatedDeltaNet; loaders: bf16 HF, MLX-4bit, ModelOpt + official NVFP4, per-tensor FP8, AWQ-int4; fp4 pack/unpack; `num_layers` truncation |
| KV | Paged blocks + hash prefix cache + GDN recurrent state; engine-level prefix hits |
| Engine | Continuous batching, submit/poll, one forward per tick |
| Training | Hand-written tape + AdamW; OPD loop; JSONL pretraining with checkpoints |
| Tests | 64 hermetic (e2e, parity, gradcheck, kv, server, per-format loader) |
| Serving | OpenAI-compatible server + SSE + chat UI |

Every checkpoint format the 27B family ships in has a hermetic loader test
(bf16, MLX-4bit, NVFP4 ModelOpt + official, FP8, AWQ-int4); the remaining
risk is the real 27B weights themselves, not the format handling.

## Phase 1 — 27B real-weight bring-up

The only thing between today and the target model is the checkpoint itself.

- Download Qwen3.8-27B (NVFP4) via `HF_ENDPOINT=https://hf-mirror.com`;
  `TILERL_QWEN38_SOURCE` in config is the slot.
- Load with `num_layers` truncation first (2–4 layers), then full 64 layers.
  NVFP4 ≈ 20 GB on disk — fits this Mac's unified memory; CPU forward is slow but
  is the correctness path, Metal the perf path.
- Exit: 27B generates coherent text on CPU and Metal; NVFP4 dequant has a
  parity check against the torch-eager reference; bench entry for
  prefill/decode on both targets.

## Phase 2 — CUDA target on the pod

- Needs a kubeconfig from the user (kubectl has no context today);
  `Dockerfile` + `k8s/pod.yaml` + `scripts/pod.sh` are ready.
- Run the full suite + bench under `TILERL_TARGET=cuda`. Same kernel source
  as CPU — failures are tilelang bugs, fixed upstream via PR (authorized),
  not worked around locally.
- Exit: cuda suite green, bench numbers in a wins entry, support-matrix cuda
  column filled with real results.

## Phase 3 — training on real weights

- OPD and pretrain currently run on tiny random weights. Run both on the
  27B (few layers first): loss must decrease, tape gradcheck must pass on
  the real architecture (every param gets a finite grad).
- Replace torch-eager backwards with tilelang kernels where the bench says
  it matters — each is tagged `# ponytail: torch-eager backward, tilelang
  kernel when perf demands`; each replacement lands with the gradcheck gate.
- Exit: one OPD run and one pretrain run on 27B weights with decreasing
  loss; a bench entry for train step time.

## Phase 4 — serving scale (PD / AFD)

Design only today (`docs/design-pd-afd.md`); the seams already exist (block
tables, separate KV/GDN stores, layer-typed model).

- PD disaggregation first: one KV+state handoff per request, engine seam is
  role-agnostic. Missing: KV transport, router, D-side admission.
- AFD after, only once a single engine is batch-limited: per-layer
  activation transport + micro-batch pipelining.
- Exit: a PD pair serves a request end to end; handoff cost benchmarked.

## Parked (trigger-based, not scheduled)

- **GDN2** — clean-room implementable, license blocks copying. Trigger: a
  GDN2 checkpoint appears in the target family. Assessment in
  `docs/design-gdn2.md`.
- **fp8 / sm100 / rocm cells** — registry slots exist; trigger: hardware in
  hand.
- **AFD** — trigger: single-engine batch saturation (Phase 4).

## Standing discipline

- Every hot-path change ships a bench entry (`docs/experience/wins/` or
  `errors/`).
- New op ⇒ parity check; new backward ⇒ numerical gradcheck. No exceptions.
- LOC is the thing to cut; a shorter diff passing the same gates wins.
