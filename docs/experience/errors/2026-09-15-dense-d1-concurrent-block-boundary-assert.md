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

## Root cause (bisection hypothesis — UNVERIFIED)

`DraftHead.step` (`spec.py`, the assertion around line 407) plans each row's
draft write against the blocks it currently owns:

```python
dblocks = r.draft_blocks if r.draft_blocks else r.blocks
assert len(dblocks) * BLOCK_TOKENS > hi   # hi = r.seq_len - 1
```

At `seq_len=929` the trunk needs 59 blocks for position 928, but the row still
holds 58 (58×16 = 928, and the check is strict), while `draft_pos=926` puts
the planned write exactly at that position. The hypothesis: when four dense
rows decode in one tick and several cross a 16-token boundary together, the
block that grows the row is allocated (or expected to already exist) on a
path that assumes it was added before the draft step — under one-row-per-tick
or non-concurrent timing the trunk growth lands first, but a multi-row tick
reaches `_draft.step` with a row whose trunk block for the new boundary has
not been attached to the set the draft plans against. The trunk and the draft
disagree by one block at the crossing, and only when the rows align.

This is a hypothesis from the assertion numbers and the concurrency needed to
reproduce; the exact ordering between trunk growth and `r.blocks`/
`r.draft_blocks` visible to `step` has not been traced. A correct fix must
reason about draft position versus the blocks the row will own AFTER the
tick's growth, not the ones it entered with, and should cover the dense
trunk-reuse path (`r.blocks`) and any draft-owned path (`r.draft_blocks`)
symmetrically.

## Fix (named, not landed)

`src/tilerl/spec.py` `DraftHead.step` block planning together with the dense
decode growth in `src/tilerl/engine.py`: reconcile draft-position with
row-owned-block accounting so four dense d1 rows crossing a 16-token boundary
in the same tick either grow before the plan is built or plan against the
post-growth block count. Queued for the dense-engine owner after step 11
(`spec.py` may overlap the sparse `SparseRuntime` work, so it is sequenced
after to avoid editing the file twice).

A gate needs a deterministic CPU/tiny reproduction, not another ~1/3
real-card warmup: build four dense rows with d1 and drive one `step()` that
lands all four at the `(58×16 → 59×16)` boundary, asserting the tick
completes and the writes land on owned pages. Red today on the alignment,
green after.

## Rule

An off-by-one block check that only fires when N rows align at a boundary in
one tick is a concurrency-ordering defect, not a capacity defect: size-based
single-row probes pass forever. The repro must force the multi-row alignment.

## Deterministic gate

The ~1/3 card warmup is the bug report, not the gate. The acceptance gate is a
deterministic CPU/tiny test that forces the four-row alignment the card only
hits by timing. Lives in `tests/test_e2e.py` beside the dense spec suite
(reuses `_random_draft`, `_drain`, `tiny()`, `RefBackend`).

Build knobs — dense d1 (the bug is the dense trunk-reuse path `r.blocks`, not
the sparse-owned `r.draft_blocks`):

```python
cfg = tiny()  # 2 layers, idx 0 full-attn / idx 1 gated; BLOCK_TOKENS = 16
engine = build_engine(cfg, build_random(cfg, seed=7), get_backend(),
    num_blocks=32, num_slots=4, max_batch=4, max_total_tokens=4096,
    draft=_random_draft(cfg, 7, model), spec_depth=1, sparse_k=0)
```

Force the crossing: every prompt is `L = k*16 + 1` tokens (`L = 49 = 3*16+1`
fits the pool). After prefill a row sits one token past a block boundary, so
its first decode drafts at `hi = seq_len - 1 = k*16`, which owns `k+1` blocks
under the strict `len(blocks)*16 > hi` check — the one-block-short condition.

Submit all four BEFORE draining and drive the engine with `step()` directly
so the rows batch in one tick instead of racing on poll timing:

```python
prompts = [np.arange(7, 7 + L) + i * 1000 for i in range(4)]  # distinct, equal length
rids = [e.submit(p, SamplingParams(temperature=0, max_new_tokens=4, seed=i))
        for i, p in enumerate(prompts)]
```

Then step until the alignment precondition holds, and make the NEXT step the
probe — this is what turns ~1/3 into every run:

```python
for _ in range(N):
    e.step()
    live = [r for r in e._running]
    if (len(live) == 4
            and all(r.phase == _PHASE_DECODE for r in live)
            and len({r.seq_len for r in live}) == 1):
        break
e.step()  # the tick that drafts all four at hi == k*16
```

Red today: that step raises the `spec.py` AssertionError (`pytest.raises`
marks it red pre-fix; one tick 500s every running request). Fixed asserts:

```python
# no exception; rows still running, not failed by the tick
for r in e._running:
    assert len(r.blocks) * BLOCK_TOKENS > r.seq_len - 1   # owns the block it drafted
outs = [_drain(e, [rid], 4)[rid] for rid in rids]
assert all(len(o) == 4 for o in outs)                     # all four complete
assert e._kv.free_blocks == free_before                  # zero block leak
```

Expected fixed behavior: before `_draft.step(rows)` plans, each dense row has
grown — or the plan is computed against — the blocks covering
`seq_len - 1 + depth` for the tick, so draft position and the post-growth
owned-block count agree for four rows aligned at the boundary in one batch.

**Sparse symmetry note:** the same plan logic has a `r.draft_blocks` branch
(sparse rows own a separate dense draft pool). When that code moves in the
post-step-11 work, give it an equivalent forced-alignment gate (sparse
`sparse_k>0` build, four sparse rows at a `k*16+1` crossing) rather than
assuming the dense gate covers it — the card failure was dense, but the
one-block-short arithmetic is written in the shared function.
