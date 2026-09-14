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
