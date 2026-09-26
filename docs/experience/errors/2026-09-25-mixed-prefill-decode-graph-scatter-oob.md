# Mixed sparse prefill+decode crashes captured graph replay with aten ScatterGather OOB — V100, 2026-09-25

## Trigger

Production c05969b1, sm70 V100, sparse_k=128, sparse window 1024 tokens (64
pages), decode graph on, depth-1 draft, 4 slots, cold 8 GiB. While one row is
streaming a decode answer, a second long prompt (5.2k tokens) is submitted.
The long prompt's chunked prefill shares ticks with the decode row
(`dec=1 pre=1` mixed eager ticks). When the long prompt FINISHES, the next
tick is the decode row alone on a **captured sparse decode graph**, and that
replay crashes the CUDA context:

```
aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144
Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed
```

Dozens of lanes fail contiguously (thread 0..64). The non-blocking Python
stack surfaces later in `_verify -> _sample_batch -> torch.tensor(hot, ...)`
or `finalize -> page_bounds_one`; these are poisoned-context reporting points,
not the faulting kernel. `health` keeps answering from the loop while the
context is dead (the separate fatal-exit fix, phase805, is what turns this
into a clean process exit).

## What is established (elimination list — do not re-walk these)

All measurements are the same 5.2k-prompt concurrent scenario.

1. **Eager never crashes; only captured graph replay crashes.** A run that
   takes the eager path on every tick (179 path=eager, 0 graph) completes.
   Crashing runs have 60–140 graph ticks. Do not infer ordering from an
   "assert fixed it" run without confirming the graph was actually used.
2. `CUDA_LAUNCH_BLOCKING=1` does **not** fix it (clean tree still fails, 65
   asserts). Not a plain async launch race.
3. An in-graph **dependency-only read** of the index tensors (read
   col/nsel/sel, write a guaranteed-zero derived value into `table`, no
   predicate) does **not** fix it. Not a missing intra-graph stream/event
   edge that an extra read dependency would close.
4. Clamping the python-visible `_select_device` indices does not fix it:
   `safe_pos` into `cand_idx`/`s_l2p` gather, scatter `col` into `k+own_w`,
   and `phys` lower bound, each and all together — still 65.
5. A device **data_ptr/shape trace** across every static tensor the graph can
   read — SparseForward staging (own_table/page_base/cand_idx/own_log/
   own_valid/n_cand/win/own_len_t/s_l2p/s_bounds), per-rid tracker
   l2p_t/bounds_t/bounds_valid, graph ids/pos/sl/slots/sql/logits, trunk
   K/V pools + scales, GDN states/step_states/conv_windows/win_parity, and the
   draft K/V pool — records zero moves across the mixed ticks before the
   crash. Not a tensor reallocation the graph holds a stale address to.
6. A synchronize-then-host **content assertion** over every fed-in index
   against the real target size — ids<vocab, slots<num_slots,
   own_table<num_blocks, s_l2p∈[-1,num_blocks), n_cand≤cmax, per-row draft
   block ids in the draft pool — is all clean at the crashing replay. The
   bad index is not present on any tensor fed into the replay.
7. CPU invariant sweeps across the mixed sequence (phys in
   [0,num_blocks), refcount>0, l2p==resident, no double-owned phys) are 0
   errors; the CPU cell has no device bounds check and cannot reproduce this.
8. `compute-sanitizer memcheck` is unusable here: it breaks TileLang's nvcc
   JIT (`tilelang_callback_cuda_compile` RuntimeError at the first new
   shape). Do not spend a window on it.

## Where the faulting op is

`CUDAGraph.debug_dump` of the captured B1 sparse decode graph shows, per
layer, one `at::native::_scatter_gather_elementwise_kernel<128,4, ...Op...>`
immediately preceded by an arange and an `_Indexing_cu masked_fill_kernel`:
that sequence is `_attention_args_device` (`sparse_engine.py`)

```python
col = nsel[:, None] + torch.arange(self.own_w, device=...)
own_phys = own_phys.masked_fill(~self.own_valid, 0)
table.scatter_(1, col, own_phys)          # table width = k + own_w
```

16 scatter_gather nodes (one per full-attn plane) + 1 embedding node
(`embedding_f16_kernel<<<2,64>>>`). The contiguous-lane signature matches
this elementwise scatter. The graph replays the *recorded* ops; `col`/`nsel`
are recomputed from staging each replay inside the graph, so a python clamp
on the returned tensor and a host content check cannot see or bound the
in-graph value.

## Root cause (confirmed, fix landed)

The faulting op is the per-plane `table.scatter_(1, col, own_phys)` in
`SparseForward._attention_args_device`. `col = nsel + arange(own_w)`
(own_w = WINDOW_PAGES+1 = 65 at sparse-window 1024); the 65 failing lanes
(thread 0..64) match those 65 columns exactly.

`nsel` was **correct at the select output but garbage at the attention
consume point inside the same replay**. A device probe counted
`(nsel > k)` right after `order_members` = 0, while a col-OOB counter at
the scatter accumulated ~520 per replay and the raw `nsel` read back as
±3.7e18/−5.9e18. Clamping nsel to [0,k] alone eliminated the crash.

The cause is storage lifetime under CUDA graph capture: `_dphys`,
`_dnsel`, `_dchosen` cached tensors **lazily allocated during the first
captured forward** — graph-private memory-pool storage. The fill()
staging tensors the design already treats correctly (`s_l2p`,
`s_bounds`) are allocated in `__init__`, **outside** capture. After a
mixed eager tick changed the graph's allocation history, replaying read
the baked-in private addresses as garbage, so nsel ran out of range.

**Fix** (`sparse_engine.py`): allocate persistent per-group
`s_nsel`/`s_phys`/`s_chosen` `[n_groups,B,k]` in `__init__` (outside
capture, like `s_l2p`), and inside `_select_device` `copy_` each tick's
result into the slice for the group, returning the stable slice. The
recorded downstream scatter/gather now bakes a stable address.

Verified on V100 sm70, production config (c05969b1, sparse_k=128,
window 1024, decode graph, depth-1 draft, 4 slots, cold 8 GiB): before
the fix the concurrent stream+5.2k-prompt scenario crashed with 60-65
ScatterGather asserts; after the fix (no clamp) 3 consecutive rounds =
0 asserts, 538 captured decode forwards, every client stream completed
err=None. CPU regression:
`tests/test_sparse_graph_ptr_stable.py` snapshots the three new buffers
and asserts the cached select outputs ARE the persistent slices
(fails on the pre-fix tree, passes after).

## Probe traps that wasted rounds (do not re-walk)

- A post-`replay()` host read of a device accumulator never prints after
  the crash: the device-side assert poisons the context and the
  synchronize before the read raises first. Read-out must be on the
  replay BEFORE the faulting one, or the clamp must keep the bad replay
  alive.
- A `tracker`/`self.tracker` NameError in capture-time setup silently
  forced eager fallback — an "assert fixed it" run with zero graph ticks
  proves nothing. Confirm a graph bucket was actually built and check the
  forward counters before trusting a clean run.
- Clear `__pycache__` on the card after rsync; equal-mtime edits can
  leave stale bytecode and a probe that looks loaded but is not.

## Artifacts on the card

- failing-run step logs and stacks: `~/oob_probe/`, `~/oob_ptr/server*.log`
- graph debug dump: `~/oob_ptr/gdump/sparse_graph_B1_W2.txt`
- probe trees: `~/tilerl-fix-ptr` (data_ptr trace + content assert +
  OOB_GRAPH_DUMP hooks; all env-gated, not on main)
- clients: `~/smoke_stream_ts.py`, `~/longp.py`; orchestrator
  `~/taskD_out/run_concurrent.sh` (stream first, long prompt +4s)
