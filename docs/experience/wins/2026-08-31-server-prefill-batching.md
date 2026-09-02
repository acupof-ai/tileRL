# Server prefill batching + enable_thinking + 8K context — sm70, 2026-08-31

> Status: Shipped

## Context

B=8 through the OpenAI-compatible server was 2.8 tok/s vs the direct engine's
10.5 tok/s — a 3.75× gap. The decode graph was working (`graphs=[(8,1)]`), the
daemon thread was not the bottleneck. Root cause: 8 concurrent HTTP requests
arrive over ~8 ms; the daemon ran `step()` between arrivals, admitting 1–4 at a
time. Each partial batch started a separate prefill, and the rest landed in
eager mixed ticks (decode graph off, ~10× slower per tick). 17 prefill
forwards + 6 mixed forwards for 8 short prompts.

Two adjacent defects: `chat_template_kwargs={"enable_thinking": false}` was
silently ignored (Pydantic dropped the unknown field; `_render_chat` hardcoded
ChatML), so the model burned tokens on thinking. And `num_blocks=256` capped
context at 4096 tokens.

## What Worked

Three independent fixes, one short diff (23 lines):

1. **Prefill batching window** (`engine.py _loop`): when idle and requests
   start arriving, wait 10 ms before the first `step()`. A burst of HTTP
   requests accumulates and shares one prefill tick. 10 ms is negligible
   against a 264 ms decode tick or a 15 s prefill. Result: 1 prefill forward
   for 8 requests (was 17 + 6 mixed).
2. **enable_thinking** (`server.py`): accept `chat_template_kwargs` on the
   request model; when `enable_thinking: false`, append an empty thinking
   block (`<think>\n\n</think>\n\n`) to the assistant prefix — exactly what
   Qwen3's Jinja template renders. The model answers directly.
3. **8K context** (`cli.py`): `num_blocks=256 → 512` (+512 MB HBM, fits 32 GB).

B=8 server: 2.8 → 12.9 tok/s (4.6×). Wall time 168.5 → 31.4 s.

Long-context (B=1, chunked prefill, 1800s timeout):

| prompt_tok | ttft_s | decode_t/s | wall_33_s |
|---:|---:|---:|---:|
| 1042 | 61.3 | 10.8 | 64.3 |
| 2062 | 198.5 | 7.7 | 202.7 |
| 4112 | 686.8 | 3.5 | 696.0 |

TTFT scales ~O(T²) — the GDN serial scan dominates prefill (61→199→687s).
Decode at 4K drops to 3.5 tok/s (vs 7.7 at 2K): longer context = more KV
blocks to touch per decode tick, memory-bandwidth-bound on V100.

## Rule

A daemon that runs `step()` with zero batching window turns every concurrent
burst into eager mixed ticks. Wait one network-RTT-scale interval before the
first prefill of an idle period — the graph-captured decode it preserves is
worth ~10× per tick.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-31 | pre-fix | V100 sm70 | cuda | Qwen3.8-27B-NVFP4 | — | — | 2.8 |
| 2026-08-31 | post-fix | V100 sm70 | cuda | Qwen3.8-27B-NVFP4 | — | 264 | 12.9 |

Raw artifacts: `scripts/eval_b8_server.py`, `scripts/bench_long_context.py`,
server `/health` stats.
