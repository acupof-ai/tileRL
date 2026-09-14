# Four concurrent dense+d1 rows 500 at a decode block boundary

## Context

V100 sm70 (n37-002-027, 31.7 GiB), the hybrid serve (`--sparse-k 128
--sparse-min-tokens 8192 --draft model_mtp.safetensors --depth 1
--decode-graph`, four slots), found 2026-09-15 while gating the
`decode_graph.py` move (#616).

The launcher warmup `scripts/serve_warmup_hybrid.py` ends with a **4x2k**
arm: four concurrent ~2,000-token prompts, each `max_tokens=8`,
`enable_thinking=false`, fired through a `ThreadPoolExecutor(4)`. These are
DENSE rows (under the 8192 sparse threshold) with the MTP d1 draft on.

Roughly **one run in three** the arm returns four HTTP 500s; the serve log
shows one `spec.py` assertion that fails the whole tick (`step()`'s handler
fails every running request when `_run_forward` raises):

```
File ".../src/tilerl/engine.py", in _run_decode_graph
    self._draft.step(reqs)
File ".../src/tilerl/spec.py", line 407, in step
AssertionError: draft would write position 928 but the row owns 58 blocks x 16
= 928 positions (seq_len=929, draft_pos=926)
```

A single request never triggers it; dense-7k and sparse-9k warmup arms pass
every boot. The failure needs several rows in one tick at the boundary.

## Not the graph move

The first failing boot was on ff3eaa4a (#616), which made it look like a
regression from moving the captured graphs into `decode_graph.py`. A repeated
A/B across trees disproved that — the assertion is pre-existing:

| tree | content | 4x2k warmup over repeated boots/runs |
|---|---|---|
| 87869c62 | before step 9 / F19 | clean on the boots sampled |
| 3026d507 | #615, before #616 | ~2 pass / 1 fail, identical assertion |
| ff3eaa4a | #616 decode_graph move | ~2 pass / 1 fail, identical assertion |

The decode graph path (`_run_forward` gate, `_run_decode_graph`, the captured
`DecodeGraph`) is byte-identical between 87869c62 and ff3eaa4a apart from the
extraction/rename. #615's admit hunk is gated on `self._boot is not None` and
the serve runs without `--kv-store`, so it is not on this path either. The
race is older than both PRs; the small per-boot sample just hid it until a
repeat-run gate happened to count it.


## Root cause

A growth-TIMING divergence between the eager and the captured-graph decode
paths, verified against `engine.py` (2026-09-15). `DraftHead.step`
(`spec.py`) plans its writes against the blocks each row currently owns:

```python
dblocks = r.draft_blocks if r.draft_blocks else r.blocks
assert len(dblocks) * BLOCK_TOKENS > hi   # hi = r.seq_len - 1 post-commit
```

Within one decode tick the trunk commits a token (`_sample_commit` →
`_commit` → `r.seq_len += 1`) and then the draft step runs. After a commit
that accepts exactly onto a 16-token boundary, `seq_len` becomes `k*16 + 1`,
so the draft's `hi = seq_len - 1 = k*16` needs the `(k+1)`th block (the check
is strict). The two paths disagree on whether that block exists yet:

- **Eager** — after the forward it runs an explicit post-commit growth loop in
  `_run_forward` (`while len(r.blocks)*16 <= r.seq_len-1: r.blocks.append(
  alloc_block())`, immediately before `_draft.step`). The row grows to `k+1`
  blocks, then drafts. Structurally it cannot see the shortfall.
- **Captured graph** — `_run_decode_graph` calls `_draft.step(reqs)` right
  after commit with only the PRE-FORK growth, which is sized by
  `_decode_extra_blocks(seq_len, q, …)` covering `seq_len + q - 2` (the #605
  anchor-rewrite bound). Its comment claims the pre-fork loop "already covers
  the draft's furthest write" — true for the verifier chain, false for the
  post-commit `hi` on an accepted-token boundary. The row is one block short,
  `spec.py` asserts, and the whole batched tick fails.

4x2k warmup runs dense rows under `--decode-graph`, so it takes the graph
path; it surfaces only when four rows accept onto the same boundary in one
tick (single rows usually cross on a verifier-covered or eager tick). Not a
capacity bug and not the #616 extraction — both paths are identical before
and after that move; the divergence is older.

## Fix (named, not landed)

Pull the invariant "every decode row's `r.blocks` covers the post-commit
draft position `seq_len - 1` (plus the verifier tail)" out of the eager loop
into one shared planning helper — e.g. `ensure_draft_write_blocks(rows)` that
grows what is needed and finishes a row that does not fit, with the existing
growth semantics — and call it from BOTH `_run_forward` (eager) and
`_run_decode_graph` (captured) immediately before `_draft.step`. The graph
path drops its "pre-fork growth is enough" assumption, the two paths
converge, and `spec.py`'s strict check can never see a one-block-short dense
row. Queued for fixkv after step 11.

## Rule

An off-by-one block check that only fires when N rows align at a boundary in
one tick is a growth-timing divergence between two paths, not a capacity
defect. If one path grows inline and the other assumes earlier growth was
enough, the fix is a shared "blocks cover the post-commit write" helper
called from both; single-row probes never expose the missing call.

## Deterministic gate

Two layers: a pure planning unit test (the acceptance gate) plus a
device-level forced-graph smoke. The old `L = k*16+1` prefill construction is
**wrong** — a `k*16+1` prompt already holds `k+1` blocks from prefill, which
cover `hi = k*16`; there is no shortfall at admission. The defect is about
growth AFTER commit on the graph path, so the unit gate targets the shared
planner directly, not prompt arithmetic.

**1. Unit gate (CPU/tiny, no forward, no graph)** — in `tests/test_e2e.py`
beside the dense spec suite (reuse `tiny()`, `_random_draft`, RefBackend):

```python
# four dense+d1 rows frozen in the PRE-growth state at an accepted boundary:
# each row has committed onto k*16 (seq_len = k*16+1) but still owns only k
# blocks; the planner must grow every row to k+1.
rows = make_rows(n=4, blocks_per_row=k, seq_len=k*16+1, sparse_on=False)
kept = engine.ensure_draft_write_blocks(rows)          # the extracted helper
assert len(kept) == 4
for r in kept:
    assert len(r.blocks) * BLOCK_TOKENS > r.seq_len - 1
# a row the pool cannot fit is finished, not raised; pool bookkeeping restored
```

Red today: the graph path has no such helper (the precondition only holds on
the eager inline loop), so exercising the equivalent logic on aligned rows
raises the `spec.py` assertion. Green after the helper is extracted and both
paths call it. Build knobs for the engine used here:

```python
cfg = tiny()  # BLOCK_TOKENS = 16
build_engine(cfg, build_random(cfg, seed=7), get_backend(),
    num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096,
    draft=_random_draft(cfg, 7, model), spec_depth=1, sparse_k=0)
```

Fixed-state assertions are deterministic because the rows are constructed at
the boundary explicitly — equal block count, `seq_len = k*16+1`, dense — not
scheduled into alignment. Submit four equal rows only when asserting the
call-site wiring; the core guarantee is the planner's, which takes rows
directly.

**2. Device smoke (forced graph)** — V100 sm70 (pre-authorized), run the
server with `--decode-graph` and drive four concurrent dense d1 rows that all
accept onto a 16-boundary in one tick (the 4x2k warmup shape), deterministically
by equal-length prompts submitted together; assert all four complete (no 500),
each drained to `max_new_tokens`, and `free_blocks` returns to its pre-tick
value. This is the end-to-end confirmation that both paths now share the
helper. sm90/H20 parity is **deferred (H20 unavailable by decision
2026-09-14)**.

**Sparse symmetry:** the same plan function has a `r.draft_blocks` branch
(sparse rows own a separate dense draft pool, reserved at admit). When that
code moves post-step-11, give the planner a sparse-row unit case (admit
reservation vs post-commit end) rather than assuming the dense gate covers it.

## Fix (landed 2026-09-15)

`Engine.ensure_draft_write_blocks(rows)` in `src/tilerl/engine.py` is the one
post-commit growth planner: for each decode row it grows `r.blocks` until they
strictly cover `seq_len-1` (the dense draft's `hi`), finishing a row the pool
cannot fit as `pool_exhausted`; a sparse row is bound-checked against its
admit-reserved `r.draft_blocks` (verifier tail) and never trunk-grown. Both the
eager path (`_run_forward`) and the captured-graph path (`_run_decode_graph`)
call it immediately before `_draft_step`; the graph path's "pre-fork growth is
enough" assumption is gone. The eager inline post-commit loop is deleted.

Gates `tests/test_dense_d1_growth.py` (4), constructed rows, no forward:
q=1 four-row boundary grow; q=2 two-token verifier-tail row (post-verify
`seq_len = k*16+2`); pool-exhaustion finishes the short row alone and restores
its blocks/slot; sparse row uses reserved draft_blocks with zero trunk growth.
The pre-fork verifier-tail coverage is unchanged — the planner only adds the
boundary-crossing block the anchor math (`seq_len+q-2`) misses. CPU: e2e 81,
sparse 62, rl 33 (on-policy graph/recapture), server 73, decode-graph/dflash
29, all green. Device forced-graph smoke (the 4x2k warmup shape) runs on the
V100 after merge by ops.
