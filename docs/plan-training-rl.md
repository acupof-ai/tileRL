# Training / RL roadmap to SOTA — TP, CP, long context, one runtime

Status 2026-08-28. Serving is at 90.9 tok/s B=1 on one H20 (Arle 84.5);
training exists only as the tape + `train_step` on the tiny model. This is the
plan to make the SAME runtime train the 27B: on-policy distillation (OPD /
self-OPD with an EMA teacher, as `agent-infer/crates/train`), full-parameter
fine-tuning, and 32k–256k contexts across the 8-card pod. Numbers first.

## The physics (8 × H20, NV18 all-to-all NVLink, 96 GB each, NCCL 2.28)

| quantity | value | consequence |
|---|---|---|
| bf16 weights (dequantized dump) | 42 GB | one card cannot hold weights + optimizer |
| full-param mixed precision (bf16 w + fp32 master + 2 Adam + bf16 grad) | ~336 GB | needs ≥4-way sharding (ZeRO-1/TP); 8-way = 42 GB/card + activations |
| LoRA on frozen fp4 base (r=64 on every linear) | ~0.5 GB adapter + 22.8 GB base | one card; adapters train, base stays the served bytes |
| H20 bf16 tensor peak | ~148 TFLOPS | training is compute-bound here, not bandwidth-bound |
| 6·N FLOP/token @ ~21B active | 126 GFLOP/token | 50% MFU ⇒ ~590 tok/s/card ⇒ ~4.7k tok/s on 8 cards; a 32k sample ≈ 7 s |
| KV for one 32k sequence (16 full-attn layers, 8 KV heads, D=256, bf16) | 4.3 GB | 32 concurrent 32k rollouts = 137 GB ⇒ KV must shard (TP-8: one KV head per card) |
| rollout decode today | 90 B=1 / 309 agg B=8 | a 32k rollout at B=1 = 6 min ⇒ RL time is rollout time ⇒ batch ≥32 decode is a training lever |

## Phases (each: files, gate, effort; a phase ships only with its gate green)

### P0 — the tape is complete on sm90 (1–2 agent-days)
Today five backwards are torch-eager (`rope_bwd`, `attention_bwd`,
`linear_attn_bwd`, `silu_mul_bwd`, `embedding_bwd`) and the 27B cannot even
allocate fp32 masters. Deliver:
- `linear_fp4` / `linear_fp8` backward = STE: grad-x through the dequant GEMM
  (bf16 WGMMA, K-major transpose), grad-w only for trainable masters.
- Dense attention fwd/bwd as TileLang (SOTA copy: tilelang
  `examples/flash_attention/example_mha_bwd.py`), GDN chunk backward
  (`example_chunk_delta_bwd`), fused rmsnorm/silu/rope backwards.
- Gate: numerical gradcheck on tiny (exists) + per-op backward parity at 27B
  dims (`op_parity.py` grows a `--bwd` arm); a `train` harness suite on sm90
  with tok/s. Bench entry.

### P1 — 27B trains on ONE card: LoRA-OPD (2–3 agent-days)
Mirror `agent-infer/crates/train/{lora,loss,ema_self_teacher}.rs`:
- LoRA adapters on every linear (rank/alpha from config), fp4 base frozen and
  served as is; decode gains two small GEMVs per adapted linear (in-kernel
  epilogue later). Serving and training share the weights by construction —
  no weight sync, no second product line.
- OPD loss (reverse-KL to teacher logits over the student's own samples);
  self-OPD with the EMA adapter as teacher; snapshot/restore of {student
  adapter, EMA adapter, AdamW moments} as ONE unit (their R2 lesson).
- Gate: loss falls on a held prompt set; MMLU of the adapted model does not
  drop vs base (the 0-shot runner `scripts/mmlu.py`); train tok/s and
  rollout tok/s recorded.

### P2 — decode at B=32–64 for rollouts (2 agent-days)
The RL loop is rollout-bound. Extend the tensor-core decode GEMM from MX=8 to
MX=32/64 (A tile = 16 rows ⇒ 2–4 mma per k-tile, same B fragment), split
attention already scales with B. Gate: harness `decode-kv` B=32 row ≥ 3× the
B=8 aggregate; verify PASS.

### P3 — TP across the pod (3–4 agent-days)
Column-parallel qkv/gate_up/in_proj, row-parallel o/down/out_proj with one
all-reduce per sublayer (64 layers × 2 ≈ 128 all-reduces/tick; at ~10 µs on
NV18 that is ~1.3 ms — acceptable for decode, negligible for training).
Attention TP = one KV head per card (8 heads / 8 cards) ⇒ the 32k KV
problem is solved by the same cut. GDN TP = value heads split (48 / 8).
- **Collectives, two layers behind one `tilerl/ops/comm.py` seam.** Training
  traffic (grad all-reduce, ZeRO shards, CP ring: MB–GB) goes through
  `torch.distributed` NCCL — the "torch is a container" rule bars autograd
  and optim, not transport, and nothing else is as mature. Decode TP traffic
  is 10 KB per all-reduce, 128 times per tick: NCCL's ~15 µs floor would be
  ~2 ms of a ~3 ms TP-8 tick, so those go through a TileLang CUDA-IPC
  one-shot all-reduce kernel (each card writes its slice into peers' mapped
  buffers, one kernel reduces; ~3–5 µs, graph-capturable by construction,
  later fusable into the GEMV epilogue — the vLLM / TensorRT-LLM design in
  our one backend). Alternatives priced and declined: raw NCCL via ctypes
  (same latency, re-implements rendezvous), NVSHMEM (only pays for MoE
  all-to-all; this model has none).
- Gate: TP-8 decode tick vs TP-1 (expect ~1.4 ms + weights/8 ⇒ B=1 tick ≈
  3 ms, >300 tok/s single stream); loss bit-identical to TP-1 on tiny
  (deterministic reduce order); harness rows per TP degree.

### P4 — CP for long-context forward/backward (4–5 agent-days)
Sequence sharded across ranks, weights replicated (DP-style grad all-reduce),
`CpContext::single()` byte-identical single-card path (their design).
- Full-attention layers: ring attention (pass K/V shards around the ring,
  online-softmax merge — the split-KV combine we already have IS the merge).
- GDN layers: the gated-delta recurrence is linear in the state,
  `S_i = A_i S_{i-1} + B_i`, so chunk states compose: each rank computes its
  local (A, B) over its shard, one all-gather/scan across ranks fixes the
  incoming state, second pass computes outputs — a parallel prefix, not a
  serial hand-off. Backward is the same scan reversed.
- Gate: 256k-token forward+backward on 8 cards with per-card activation
  O(seq/8); gradients match the single-card path on a 32k sample to 1e-3;
  tok/s per card vs the compute ceiling above.

### P5 — full-parameter training (2–3 agent-days)
ZeRO-1 optimizer sharding (each rank owns 1/8 of the fp32 masters + Adam),
bf16 params + grads everywhere, re-quantization of served fp4 blocks after
each step (block scale + nibbles, the twiddled layout) so serving never
sees a bf16 copy. Gate: memory per card ≤ 60 GB at 32k; MFU ≥ 40%; MMLU
non-regression.

### P6 — SOTA comparison (1 agent-day, recurring)
Same pod, same model, same task: rollout tok/s, train tok/s, time per RL
round vs verl / OpenRLHF / slime (whichever load the 27B here). Recorded like
`docs/experience/2026-08-28-vs-sglang-h20.md`; the numbers set the next
round's targets.

## What torchtitan does, and what of it transfers

torchtitan (pytorch/torchtitan) composes FSDP2 (per-parameter sharding),
TP (DTensor `ColwiseParallel`/`RowwiseParallel`/`SequenceParallel`, loss
parallel, async TP micro-pipelining), PP (`pipeline_parallel.py`), CP
(`distributed/context_parallel/api.py` wrapping
`torch.distributed.tensor.experimental._context_parallel_shard`) and EP,
in a fixed order — TP → activation checkpointing → compile → FSDP — over one
device mesh (`ParallelDims`), with float8/MXFP8 training, a seed checkpoint
loaded by FQN so any shard can initialize itself, and a `cudagraph.py` that
captures the whole fwd+bwd step with static parameter buffers.

Transfers directly (no torch autograd/DTensor needed):
- **Mesh-first**: one `ParallelDims`-style object (dp, tp, cp, pp) that every
  op reads its rank/coord from — agent-infer's `CpContext` is the same idea.
- **Fixed application order** and "model forward is a loop over blocks":
  our `Model.forward` already is; keep TP/CP decisions inside ops, never at
  the top level.
- **CP as sequence sharding of inputs/labels/positions with load balancing**
  — their `_HeadTailLoadBalancer` gives each rank a head chunk and a tail
  chunk so causal attention work is even; adopt it (a rank's shard =
  `[i, N-1-i]` chunks), for both ring attention and the GDN scan.
- **Seed checkpoint by name**: shards initialize from one checkpoint by
  parameter name — our `load_hf` key map is that.
- **fwd+bwd under one CUDA graph** with static weight buffers: our decode
  graph does this for inference; the training step should too (P1 gate).
- **float8 training**: our fp8 GEMM path (`linear_fp8`, WGMMA) is the
  forward half; the backward GEMMs in fp8 are P5 material.

Does not transfer: DTensor and `fully_shard` (they are torch.autograd
machinery — our tape owns backward), `torch.compile`, and their ring
attention (SDPA-specific); we write ring attention and the GDN scan as
TileLang kernels on top of the comm seam.

## Order and why

P0 → P1 first: they make the 27B trainable on the hardware we have with no
new parallelism, and P1 is the product (agent-OPD). P2 is the RL-time lever
and is pure kernel work we already know how to do. P3 before P4: TP is the
simpler cut, it also fixes rollout KV memory, and CP reuses TP's comm seam.
P5 needs P3's sharding. P6 runs after each of P1/P3/P4 to keep the targets
honest.

## Risks named now

- Collectives inside captured decode graphs (P3): NCCL calls capture into
  CUDA graphs only with the right stream discipline; fallback is
  graph-per-layer.
- The e2m1 grid after full-param updates (P5): re-quantization noise per
  step; gate it with MMLU, not loss curves.
- CP for GDN (P4) is the novel piece; the scan formulation must be tape
  gradient-checked on the tiny model before any 27B run.
- Host contention on the pod (other tenants) — every perf gate is
  loadavg-stamped and only counts on a quiet host.
