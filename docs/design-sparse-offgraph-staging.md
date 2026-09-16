# sparse captured decode: device-resident off-graph staging (D-0 → D-3) — design, 2026-09-16

> Status: D-0 shipped with CPU byte-equal coverage (no perf claim until H20
> reopens); D-1..D-3 designed here, pending device acceptance after the H20
> freeze lifts.

## Context

#678 removed the per-tick finalize host page readback from captured sparse
decode (32k 33.43 → 27.55 ms/fwd, D2H 109 → 6.3). nsys after #678 still
shows ~6 D2H + ~26 `cudaStreamSynchronize` per forward and ~2.4 ms of GPU
kernels inside ~27.5 ms: the remaining stall is the work that runs **outside**
the captured graph around `SparseDecodeGraph.run` — input staging, verify chain
construction, sampling and the draft step. This is the plan to close it without
re-introducing host syncs, one independently-verifiable stage at a time.

The dense graph already solved the input half
(`DecodeGraph`, int32 static device buffers + pinned host staging +
`copy_(non_blocking=True)`); sparse still allocated `torch.tensor(list,
device=)` per field per row each tick.

## Stages

### D-0 — sparse decode input staging static + pinned (this PR)

`SparseDecodeGraph.run` used to build, per row per tick:
`torch.tensor(chain, device=)`, a device `torch.arange` for positions, and four
scalar device assignments. D-0 adds per-SparseDecodeGraph persistent **host
staging** tensors (`_ids_h/_pos_h/_sl_h/_slots_h/_sql_h`, pinned on CUDA, plain
CPU elsewhere) and the matching static long device buffers already present;
the fill logic is the static, CUDA-free method
`SparseDecodeGraph.fill_staging(...)`, followed by five `non_blocking`
`copy_()` calls and one replay.

- dtype stays **long**: sparse packed tables and the candidate/l2p gathers
  index long; an int32 end-to-end pass over the sparse kernels is a separate
  lever (the dense graph is int32 already, so the pattern is proven, but it
  needs a sparse-kernel dtype audit and a card).
- `fill_staging` is pure host indexing with no CUDA import, so its ids/pos/
  scalar/pad values are byte-for-byte asserted on CPU
  (`tests/test_decode_graph.py::test_sparse_decode_staging_fill_is_byte_equal_on_cpu`).
  The `CpuSparseGraph` eager twin and every eager path are unchanged.

Not in D-0 (intentionally split, no device to verify): `SparseForward.fill()`
still builds `own`/`resolve`/`cand` lists into device tensors per tick. That is
D-0b — persistent pinned staging for `cand_idx / s_l2p / s_bounds / own_table`,
mirroring the input staging; it touches the selection gather shapes and should
land with an H20 D2H recount.

### D-1 — device-resident draft token, no chains `.tolist()` on the graph path

`spec.py` verify builds host `chains = [[int(t)] for t in tok[:,-1].tolist()]`
(spec.py:556-558) and the DraftHead step reads tokens back. Add a device long
field on the request (`r.draft_tok_t`) holding `tok[:,-1]`; on the graph path
the chains buffer is sourced from that device tensor (D-0's staging copies it
without a round-trip). Eager and CPU keep the host list. Acceptance: graph and
eager sampled chains are token-equal over a long decode; CPU tiny stays exact.

### D-2 — sampling/verify off the host path

`engine.sample_commit` reads logits to host (`toks.tolist()`, engine.py:2304)
every tick. Keep sampled ids on device for the next tick's input (feeds D-1's
`draft_tok_t`) and only D2H the tokens a finished/detached request actually
returns. Verify acceptance mask similarly stays device until a reject must be
surfaced. Acceptance: spec acceptance rate unchanged, tokens byte-equal.

### D-3 — collapse the residual sync boundary

After D-0..D-2 recount nsys: any remaining `cudaStreamSynchronize` should be
the prefill/refresh boundary or the response D2H, not the steady decode tick.
Target is the dense graph profile at the same context (~25 ms wall, kernels a
large fraction, D2H ~2/forward). This stage is whatever the recount names; it
is deliberately not pre-specified.

## Guardrails

- One stage per PR; each needs CPU byte-equal parity plus, after H20 reopens,
  an nsys D2H/sync/GPU-busy recount vs the prior stage (put both arms in one
  process/driver, `scripts/perf2_nsys_ticks.py`).
- No default flips; eager/refresh/prefill and the CPU cell stay on host lists.
- int32 for sparse only after a sparse-kernel dtype audit.

## Rule

A captured forward surrounded by per-tick `torch.tensor(list, device=)` inputs
is still paying synchronous H2D; move the inputs to persistent pinned staging
copied non-blockingly (the dense graph's pattern), and keep the fill function
CUDA-free so its values are checkable on CPU before trusting it on a card.
