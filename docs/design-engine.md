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
- **Storage owns three things**: paged KV, GDN state, prefix cache. The engine
  asks for prefix hits and block tables; it never touches KV memory directly.
- **The model is backend-neutral**: no TileLang/torch calls above `ops/`.
- **Prefix sharing is COW, not copy**: shared blocks are read-only until a
  sequence diverges (`cow_for_append`).

## PD / AFD

Disaggregation design lives in [design-pd-afd.md](design-pd-afd.md): the seams
above (block tables, separate KV/GDN stores, role-agnostic engine) are the
extension points. Trigger: a single engine is batch-limited (roadmap Phase 4).
