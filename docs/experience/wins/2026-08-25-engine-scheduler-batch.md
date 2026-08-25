# Mixed-batch chunked-prefill scheduler + fixed-width block table — cuda/sm90 + cpu, 2026-08-25

> Status: Shipped

## Context

The engine scheduled one phase per tick (prefill OR decode, never nested) and
sized the paged-attention block_table to the widest batch member
(`max(len(r.blocks))`). Two failure modes followed:

1. **Prefill starvation.** Concurrent submits serialized — decode ticks ran
   first, so a new prompt waited for every running decode to drain, and the
   slot pool exhausted (the B=8 concurrent-submit bench crashed).
2. **Per-tick tilelang recompiles.** tilelang bakes the block_table width
   (`Mb`) and the sequence dim as compile consts. Every time a decode request
   grew into a new block the block_table widened, and paged_attention /
   write_tokens recompiled — 5 recompiles in 30 decode ticks, ~1s/tick eager
   decode on CPU.

## What Worked

**Scheduler: continuous batching with chunked prefill, copied from SOTA.**
Mirrors agent-infer's `build_forward_plan`
(`crates/infer-core/src/planner.rs`): waiting/running queues, a per-tick
token budget, decode rows first plus at most one prefill chunk sharing one
mixed forward — with vLLM's `max_num_batched_tokens` vocabulary
(`StepLimits.max_num_batched_tokens`, default 512). No preemption/swap/
recompute day-1. `step()` splits into `_build_plan()` (admit one waiting
request under `max_batch`; take all running decodes; one prefill chunk sized
`budget - len(decodes)`) and `_run_forward()` (left-aligned padded rows —
decode row = 1 token at t=0, prefill row = the chunk; per-row `seq_q_lens`
drives the kernels).

**Per-row `seq_q_lens` through the three state-touching kernels.**
`paged_attention` (per-row history = `seq_lens - seq_q_lens`), `write_tokens`
(per-row write window `[seq_len-seq_q, seq_len)`, padding positions skipped),
and the GDN chunk kernel (per-row scan bound — a decode row with bound 1 has
the same Window++qkv conv semantics as the decode kernel). The model forward
pads variable T per row and drives per-row lengths; `BatchKv.seq_q_lens`
carries them.

**Fixed-width block_table.** `_make_kv` allocates `block_table` with second
dim = pool `num_blocks` (constant for the process). The kernels index only
positions bounded by `seq_lens`, so the extra width is dead storage — and
`Mb` stops changing, so decode compiles once.

CPU (tiny model, 40-token prompt + 40 decode tokens, blocks 3→5):
paged_attention compiles exactly 2 shapes (prefill T=40, decode T=1); decode
recompiles 0 times as blocks grow.

Pod (H20, slice4, eager, B=1/2/4/8 concurrent 16-token prompts, 30 timed
decode ticks): see Results.

## Rule

Copy the SOTA scheduler, don't invent one: agent-infer's
`build_forward_plan` (decode rows first, ≤1 prefill chunk under a per-tick
token budget, mixed forwards first-class) is the whole policy. And any shape
tilelang bakes as a compile const must be fixed for the life of the process
— a per-tick-growing block_table is a recompile loop, not a dynamic shape.
Mixed batches land on arbitrary M (rows × chunk); the sm90 MMA tile must snap
to a warp-partition-valid size {16,32,64,128}, not `_round_up(M, 16)`.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-25 | e9d0ce1 | Mac (CPU) | c | tiny | — | — | 0 decode recompiles (2 shapes total) |
| 2026-08-25 | e9d0ce1 | H20 pod | cuda/sm90 | slice4 eager B=1 | — | 51.1 | 19.6 |
| 2026-08-25 | e9d0ce1 | H20 pod | cuda/sm90 | slice4 eager B=2 | — | 61.2 | 32.7 agg |
| 2026-08-25 | e9d0ce1 | H20 pod | cuda/sm90 | slice4 eager B=4 | — | 66.6 | 60.1 agg |
| 2026-08-25 | e9d0ce1 | H20 pod | cuda/sm90 | slice4 eager B=8 | — | 52.5 | 152.4 agg |

Eager path (decode_graph off); B=8 aggregate 152 tok/s shows the batch
amortization the scheduler is for — weights are read once per tick regardless
of B. B=2/4 per-tick latency is noisy (contended pod). The first B=8 run
crashed in `linear_fp8` (bM=80: WGMMA Square warp partition rejects 20
rows/warp); the snap to {16,32,64,128} fixed it — a latent bug mixed batching
exposed, since prefill chunks of 80-112 tokens would have crashed too.

Raw artifacts: `<bench stdout>`.
