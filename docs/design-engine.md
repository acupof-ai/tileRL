# Engine design

The serving stack is four layers, each with one seam. The seams held through
CUDA bring-up and the 27B slice without changes — they are the contract.

## Layers

| Layer | File | Seam |
|---|---|---|
| Frontend | `server.py` | OpenAI HTTP/SSE → `Engine.submit` / `Engine.poll`. Knows tokens, not tensors. |
| Scheduling | `engine.py` | `submit(input_ids, params) -> req_id`, `poll() -> {req_id: tokens}`, `StepLimits`. Continuous batching, one forward per tick. |
| Model | `model.py` | `load_hf` (every checkpoint format) + forward. Calls backend ops only. |
| Adapter | `ops/backend.py` | `(precision, arch) → kernels` registry — see [design-kernels.md](design-kernels.md). |
| Storage | `kv_cache.py` | `PagedKvPool` (paged blocks, COW on shared prefix) + `LinearStatePool` (GDN recurrent state) + rolling-hash prefix cache. |

Training shares the stack: `train.py` drives the same `model.py` forward
through the hand-written tape (`autograd.py`), same backend ops. One runtime.

## Rules

- **The engine seam is the cost contract**: `submit`/`poll` + `StepLimits`.
  A new target implements the loop behind it; it does not bend the seam.
- **One forward per tick.** Prefill and decode are scheduled, never nested.
- **The decode tick is a captured kernel sequence, not an interpreted one.**
  Decode is memory-bound and static: the same ops, the same shapes, every
  token, for the life of the process. A static sequence repeated 10⁴+ times is
  compiled once and replayed — eager per-op dispatch is the dev/parity mode,
  never the serving path. The capture lives behind the engine seam: `step()`
  has an eager implementation (correctness, parity) and a captured one
  (CUDA graph per shape bucket, serving); the model and backend don't know
  which is running.
- **Storage owns three things**: paged KV, GDN state, prefix cache. The engine
  asks for prefix hits and block tables; it never touches KV memory directly.
- **The model is backend-neutral**: no TileLang/torch calls above `ops/`.
- **Prefix sharing is COW, not copy**: shared blocks are read-only until a
  sequence diverges (`cow_for_append`).

## Physics (what the design must satisfy)

- **Decode is memory-bound**: weights are read exactly once per token. Pack
  them once at load (fp4 on disk stays fp4); never re-pack or re-cast per call.
- **Prefill is compute-bound**: tensor cores, bf16/fp8 IO. The 3800 tok/s
  class of target is unreachable on f32/TF32 IO.
- **Everything static about a tick is paid once**: shapes, weight layout,
  launch sequence. Per-token cost is only what the token actually changes
  (KV append, state update, sampling).

## PD / AFD

Disaggregation design lives in [design-pd-afd.md](design-pd-afd.md): the seams
above (block tables, separate KV/GDN stores, role-agnostic engine) are the
extension points. Trigger: a single engine is batch-limited (roadmap Phase 4).
