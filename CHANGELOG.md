# Changelog

Central progress record. Three event classes land a line the same day, linking
the `docs/experience/` entry: **phase exit · default flip · accept-or-reject
verdict**. Newest first.

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
