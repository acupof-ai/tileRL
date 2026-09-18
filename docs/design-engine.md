# Engine design

Four layers, one seam each. The seams held through CUDA bring-up and the 27B
unchanged — they are the contract.

## Layers

| Layer | File | Seam |
|---|---|---|
| Frontend | `server.py` | OpenAI HTTP/SSE → `Engine.submit` / `Engine.poll`. Knows tokens, not tensors. |
| Scheduling | `engine.py` | `submit(input_ids, params) -> req_id`, `poll() -> {req_id: tokens}`, `StepLimits`. Continuous batching, one forward per tick. |
| Model | `model.py` | `load_hf` (every checkpoint format) + forward. Calls backend ops only. |
| Adapter | `packages/tilerl-kernels/src/tilerl_kernels/backend.py` | `(precision, arch) → kernels` registry — see [design-kernels.md](design-kernels.md). |
| Storage | `kv_cache.py`, `kv_tiers.py`, `sparse_engine.py` | `PagedKvPool` (paged blocks; shared-prefix blocks stay read-only, no copy-on-write) + `LinearStatePool` (GDN recurrent state) + rolling-hash prefix cache. The sparse cold path adds `HostKvPages` (pinned-host KV tier), `ColdSsdFile` (mmap spill, `--cold-ssd-path`/`--cold-ssd-bytes`) and `DramSnapshots` (demoted GDN state, `--dram-bytes`). |

Training shares the stack: `train.py` drives the same `model.py` forward
through the hand-written tape (`autograd.py`), same backend ops. One runtime.

## Rules

- **The engine seam is the cost contract**: `submit`/`poll` + `StepLimits`.
  A new target implements the loop behind it; it does not bend the seam.
- **One forward per tick, mixed batch.** Continuous batching with chunked
  prefill (vLLM/sglang pattern, mirrored through agent-infer's
  `build_forward_plan`): waiting/running queues, a per-tick token budget
  (`StepLimits.max_num_batched_tokens`), decode rows first plus every prefill
  that fits the remaining token budget and one width bucket, sharing the
  forward. No preemption/swap day-1.
- **The decode tick is a captured kernel sequence on the dense serving path.**
  Dense decode is memory-bound and static: the same ops, the same shapes, every
  token, for the life of the process. A static sequence repeated 10⁴+ times is
  compiled once and replayed — eager per-op dispatch is the dev/parity mode on
  that path. The capture lives behind the engine seam: `step()`
  has an eager implementation (correctness, parity) and a captured one
  (CUDA graph per shape bucket, serving); the model and backend don't know
  which is running. Two exceptions: sm70's AUTO path disables capture (dense
  capture fails there and poisons the allocator), and the hybrid long-prompt
  path below runs eager on purpose.
- **Storage owns three things**: paged KV, GDN state, prefix cache. The engine
  asks for prefix hits and block tables; it never touches KV memory directly.
- **The model is backend-neutral**: no TileLang/torch calls outside
  `packages/tilerl-kernels/`.
- **Prefix sharing is read-only, not COW**: shared blocks are never modified
  after publishing; `PrefixStore.insert` enforces that no block is written
  after sharing. Sparse builds use a second store, `SparsePrefixCache`
  (host-blob backed, content-hash namespaced): it holds published pages' host
  blobs and bounds, never the live device blocks, and the same read-only rule
  applies. A pure-sparse build auto-selects `NoPrefixStore` for the
  block-retaining pool (it cannot retain pages sparse rows do not own); sparse
  sharing is the tracker's `SparsePrefixCache`, and passing `NoPrefixStore`
  explicitly disables that too (the RL path).

## Hybrid serve (#586)

One engine serves two regimes:

- Prompts no longer than `--sparse-min-tokens` run **dense** on the captured
  decode graph and pin their whole context in the device pool (no sparse
  sharing).
- Longer prompts run **sparse** with eager ticks and the host/SSD cold tiers.
- A dense prompt that could never fit the device pin even with the pool empty
  reroutes sparse in `submit` instead of queueing on an impossible admit.

Admit therefore has two regimes, and the dense regime reserves the sparse hot
ceiling so both can coexist. The sparse side is documented in
[design-sparse-kv.md](design-sparse-kv.md); the `submit`/`poll` seam and
one-forward-per-tick loop are unchanged.

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
