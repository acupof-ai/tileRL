# Layered architecture refactor lands — CPU + V100 sm70, 2026-09-15

> Status: Shipped — full CPU suite green on the final PR's CI
> (822 passed / 21 skipped / 1 xfailed ubuntu, 816 / 27 / 1 macos), reviewer
> verified body identity, and V100 128k sparse smoke passed on 2e8612c1.

## Context

`engine.py` (3633 lines), `cli.py` (2763) and `kv_cache.py` (1919) held
scheduling, storage, assembly, training orchestration and bench ledger in
three files. Imports crossed layers both ways and five import cycles tied the
tree together, so a change to storage forced a rebuild of scheduling context
and no gate could say which layer a module belonged to. The 11-step target
was designed first and pinned in `docs/design-architecture.md` (step 0); every
later step executed that table.

## What Worked

Eleven steps landed as eight refactor PRs (#599, #601, #606, #612, #614, #616,
#618, #622), each self-contained and merged on green CI plus review:

- Layers L0–L6 with downward-only imports, enforced by `tests/test_layering.py`
  (resolves relative and absolute imports, ALLOWLIST is empty and shrink-only).
- Four new modules: `build.py` (single assembler, 572), `decode_graph.py`
  (captured graphs, 382), `kv_tiers.py` (host/SSD/boot, 991),
  `sparse_runtime.py` (sparse state + named callbacks, 688).
- `engine.py` 3633 → 2305, `cli.py` 2763 → 1080, `kv_cache.py` 1919 → 962.
- The five cycles broke in one PR (#601); dflash2 merged into spec.
- Step 11 moved 11 sparse methods body-identically (AST-normalized diff,
  reviewer-verified); SparseRuntime exposes exactly 13 context fields and four
  named callbacks (verify, sample_commit, draft_step, bump_decode_forwards),
  no kwargs seam. Engine keeps a ~40-line facade proxy so internal call sites
  and tests keep their spellings.
- Two production correctness bugs found and fixed during the move:
  sparse no-draft full-prefix hang and the phantom final-tick block (#605).

Two rules kept it safe. The design was frozen before code, including the
pre-mortem split decisions for steps 8b/9/10/11. Every moved body was reviewed
normalized for binding only (`self.` → `ctx.`); any boundary-forcing logic
change had to be a separate commit, and the one such change (GraphCapture /
named `_draft_step`) was the first commit of the final PR.

## Results

V100 (sm70) 128k single-request sparse smoke on 2e8612c1: HTTP 200,
prompt_tokens 125452, 655 sparse ticks, prefill wall 1877.6 s (~67 tok/s),
clean block/slot release. The run exceeded the 1800 s production completion
timeout (the probe raised it to 5400 s remotely; the production constant is
unchanged), so 128k sparse prefill stays outside the default-served window —
the smoke confirmed behavior identity of the move, not a serving default.

## Rule

Design the target seams and pin them in writing before the first move; gate
the layer invariant in CI with an empty allowlist; require body-identity
normalization (AST, not eyeball) for method moves; any logic change rides a
separate commit the reviewer can isolate.
