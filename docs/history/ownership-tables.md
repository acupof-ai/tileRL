# Ownership tables from completed unit plans

These work-allocation tables were in the live design docs while the units were
open. All units landed; kept here for attribution, not as a plan.

## Cost-model units (from design-cost-model.md, 2026-09)

| Unit | Owner | PRs |
|------|-------|-----|
| `Format`/`nbytes`, call sites, checkpoint faces, byte gate | cc | #458, #462 |
| `plan`, dry-run, residency, transient peak | 52 | #460, #465, #469 |
| kernel roofline, prefill rows, calibration | 5f | #457, #463, #466, #468 |
| recompute recorded numbers, delete superseded probes | 65 | #461, #470 |

## Sparse-KV units A–E (from design-sparse-kv.md, 2026-09)

| Unit | Owner | Gate |
|------|-------|------|
| A. `plan` rows `index_keys`/`kv_hot`/`kv_cold`, `--sparse-k` dry-run, `kernel_cost` indexer rows | 52 | derived == storage bytes on tiny |
| B. page-bounds scorer + selector, CPU twins, `k >= context` equals dense; then the sm70 cell and the V100 256k bench | cc | `test_sparse_equals_dense_at_full_k`; V100 tokens/s + device bytes table at 128k/256k |
| C. page `location`, demote/promote of KV blocks on `DramSnapshots`' path | 5f | a demoted page promoted reads back byte-equal; decode tokens equal with and without demotion |
| D. learned indexer in V4.1 form (page keys, source layers, window), KL warm-up recipe, indexer backward, sparse attention backward | 65 | gradcheck on tiny; warm-up recipe runs one step on tiny |
| E. sm90 cell: indexer + selector from `deepseek_v32`, bench rows with `%bound` | after the runbook, cards 0-7 | roofline table in the wins entry |
