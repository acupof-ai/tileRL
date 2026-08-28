# Training / RL roadmap — only what changes the picture

Status 2026-08-28: serving 90.9 tok/s B=1 on one H20; training is the tape +
`train_step` on the tiny model; five backwards are still torch-eager and the
27B cannot allocate fp32 masters. Below: the findings that decide the design,
then the phases as gates. Everything textbook is left out.

## What is not obvious

1. **Inference and training are opposite bottlenecks on this card.** Decode is
   bandwidth-bound (we read 22.8 GB per tick at ~64% of 3.25 TB/s). Training is
   compute-bound: H20 has only ~148 TFLOPS bf16, so 6·N FLOP/token ≈ 126 GFLOP
   ⇒ ~590 tok/s/card at 50% MFU, ~4.7k tok/s on 8 cards, a 32k sample ≈ 7 s.
   Consequence: fp4 weights buy nothing for training throughput; the levers
   are the backward kernels and MFU, not bytes.

2. **RL time is rollout time, and rollout time is a decode kernel.** A 32k
   rollout at B=1 takes 6 min; the update step for that sample is 7 s. The
   single biggest RL speedup is decode throughput at B≥32 — the tensor-core
   decode GEMM extended from MX=8 to 32/64 — i.e. a serving kernel, not a
   trainer feature.

3. **TP degree is chosen by KV memory, not by compute.** One 32k sequence
   holds 4.3 GB of KV (16 full-attn layers × 8 KV heads × 256 × bf16 × K,V);
   32 concurrent rollouts = 137 GB > one card. The model has 8 KV heads and
   the pod has 8 cards: TP-8 with one KV head per card is the cut, and it also
   shards the GDN value heads (48/8) and the weights (22.8/8 GB).

4. **torch.distributed is the wrong tool for decode TP and the right one for
   everything else.** Decode TP is 10 KB per all-reduce, 128 times per tick;
   NCCL's ~15 µs floor is ~2 ms of a ~3 ms TP-8 tick. A one-shot CUDA-IPC
   all-reduce written as a TileLang kernel is ~3–5 µs, graph-capturable by
   construction and later fusable into the GEMV epilogue. Training traffic
   (grad all-reduce, ZeRO, CP ring, MB–GB) stays on NCCL. Both live behind one
   `comm.py` seam; the crossover is measured by a microbench, not assumed;
   IPC falls back to NCCL when peers are not mappable.

5. **CP for the linear-attention layers is a scan, not a hand-off.** The
   gated-delta recurrence is linear in the state, `S_i = A_i S_{i-1} + B_i`,
   so chunk states compose: each rank computes its local (A, B), one
   all-gather fixes the incoming state, a second pass produces outputs — a
   parallel prefix across cards, with the backward the same scan reversed.
   Full-attention CP is ring attention, and its merge IS the split-KV combine
   we already ship. Our own tape is what makes this possible: a custom scan
   backward is a handler, not an autograd.Function fight.

6. **LoRA on the frozen fp4 base is what makes "one runtime" literally true.**
   The served bytes never change during RL; the adapter is the only trainable
   state, so there is no weight sync between trainer and engine — it is a
   memory-format fact, not a synchronization protocol. Full-parameter
   training (later) has to re-quantize into the twiddled fp4 blocks after
   each step, and its gate is MMLU, not the loss curve: quantization noise
   does not show in loss.

7. **What transfers from torchtitan, and why the rest cannot.** Mesh-first
   (`ParallelDims`: every op reads its coordinate, the top-level forward stays
   a loop over blocks), head/tail sequence load-balancing for causal CP,
   seed-checkpoint-by-name initialization, and capturing fwd+bwd in one CUDA
   graph with static weight buffers. DTensor and `fully_shard` are
   torch.autograd machinery and do not transfer; that constraint is also the
   freedom in (5).

8. **Measurement rules that cost a day each to learn** (the loop from
   `docs/experience/2026-08-28-decode-52-to-84.md` applies unchanged): price
   kernels in-graph or under ncu, never with an eager harness (~40 µs floor
   hid a real −22%); one timing job per host (another tenant's CPU load moved
   rows 60%); every kernel shape comes from a bounded bucket (the first
   varied-length workload compiled 662 variants with the GPU idle).

## Phases as gates

| phase | delivers | gate (numbers only) | effort |
|---|---|---|---|
| P0 | the five torch-eager backwards as TileLang; fp4/fp8 STE backward | per-op backward parity at 27B dims ≤ 1e-2; train suite tok/s ≥ 0.97× snapshot | 1–2 d |
| P1 | LoRA-OPD / self-OPD (EMA adapter teacher), {student, EMA, Adam} snapshot as one unit | loss falls on a held set; MMLU 0-shot ≥ base | 2–3 d |
| P2 | decode GEMM MX=32/64 | harness B=32 row ≥ 3× B=8 aggregate | 2 d |
| P3 | TP-8 (KV head per card), `comm.py` with NCCL + IPC | TP-8 B=1 tick ≈ 3 ms; loss bit-identical to TP-1 on tiny; IPC/NCCL crossover measured | 3–4 d |
| P4 | CP: ring attention + GDN prefix scan, head/tail balanced | 256k fwd+bwd on 8 cards; 32k gradients = single card to 1e-3; scan gradchecked on tiny first | 4–5 d |
| P5 | full-param: ZeRO-1 masters, re-quantize to twiddled fp4 each step | ≤ 60 GB/card at 32k; MFU ≥ 40%; MMLU flat | 2–3 d |
| P6 | same-pod comparison vs verl / OpenRLHF / slime | recorded like the sglang comparison; sets next targets | 1 d, recurring |

Order: P0→P1 make the 27B trainable on the hardware we have and P1 is the
product; P2 is the RL-time lever and pure kernel work; P3 before P4 because
TP also fixes rollout KV memory and CP reuses its comm seam; P5 needs P3's
sharding; P6 after each of P1/P3/P4.

## Risks that are specific, not generic

- NCCL inside captured decode graphs needs stream discipline; fallback is a
  graph per layer.
- Re-quantization noise per step (P5) is invisible in loss — MMLU-gated.
- The GDN scan is the one novel kernel; it does not touch the 27B until the
  tiny-model gradcheck passes.
