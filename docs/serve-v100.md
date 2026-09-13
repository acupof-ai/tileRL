# Serving tileRL on the V100 (sm70)

The Qwen3.8-27B-NVFP4 server on the V100 host, a 31.7 GiB single card
(n37-002-027, Tesla V100-SXM2, sm70).

## Endpoint

- Host `n37-002-027` (ssh alias `v100`), port **8000**. The port is not
  exposed externally; tunnel:
  `ssh -L 8000:127.0.0.1:8000 v100`, then `http://127.0.0.1:8000`.
- `GET /health`, `GET /v1/models`,
  `POST /v1/chat/completions` (OpenAI-shaped, model `qwen38-27b`).

## Configuration

```
tilerl serve --model qwen38-27b \
  --slots 4 --max-batch 4 --max-ctx 8192 \
  --host 0.0.0.0 --port 8000
```

- **Dense**, not sparse. Four 8192-token contexts fit in HBM: after weights
  and pool build the card has 4.86 GiB free and the pool auto-fits the
  required 2048 pages. Sparse k=128 would attend only ~2.2k tokens of an 8k
  prompt and decodes slower on this card; the dense pool is the correct
  choice at 4×8k.
- Fused projections on (serve default). Speculative decode off: measured a
  loss on sm70 (W=1 ~0.82x; block-parallel best 1.016x).
- Decode graph off: dense capture fails on sm70 and a failed capture poisons
  torch's caching allocator; the auto path forces eager on this arch.
- Prefix cache on at its default.

A request over 8192 tokens gets a clean HTTP 400
("exceeds max_total_tokens (8192)"). A client disconnecting mid-stream frees
its slot; the next request admits.

## Operations

The server runs under `scripts/_serve_v100_dense.sh` on the host: flock-guarded
single instance, restart loop (max 10), log byte-capped at 32 MiB
(`~/serve70_dense.log`). The process python is `~/venv70/bin/python`
(torch 2.5.1+cu121); deploy code with `scripts/v100.sh`.

Stop: `pkill -f _serve_v100_dense.sh` (the supervisor trap releases the GPU).

## Measured 2026-09-13 (stable config)

- **Concurrent-prefill burst:** four simultaneous ~7,400-token prompts on the
  idle server, all returned 200 in 186–210 s; peak GPU memory 27,196 MiB of
  32,514 (5.2 GiB headroom); zero CUDA OOM, zero 5xx.
- **30/10-minute soak:** four concurrent workers cycling 1k/2k/4k/7.4k-token
  prompts. 234 turns, zero errors; per-turn p50 9.16 s; host RSS 4.4–4.5 GiB
  (flat); GPU held 27,196 MiB; zero supervisor restarts.
- **p99 ~75 s is the 7.4k-prefill contention cost, not a defect:** under four
  concurrent long prefills V100 prefill throughput is the binding term; small
  prompts serve at p50 ~9 s. ttft p50 0.35 s.

## Known issue: `/health` can stall under long prefill

While a long prefill runs, `/health` can intermittently fail to respond
(client sees 000/timeout). The completion path is unaffected and retries
succeed. Cause (2026-09-13 trace): `chat_completions` / `ws_chat` call
`engine.submit` on the ASGI event-loop thread, so a request admission blocks
the loop that also answers `/health`. (An earlier stats-lock stall in
`stats()` was fixed on 2026-09-07 and is not this event.) Clients must retry
`/health`; do not treat a prefill-window stall as the server being down.

Sparse long context is a separate, limited path: 64k serves only at B=1 and
256k prefills are OOM-killed — see
[errors/2026-09-13-v100-256k-sparse-prefill-host-oom](experience/errors/2026-09-13-v100-256k-sparse-prefill-host-oom.md).
