# Sparse window and refresh interval are settable at build — V100 sm70, 2026-09-24

> Status: `pending-remote` — the flags and their gate are landed; the device
> table below is filled in by the W×R sweep they exist to serve.

## What changed

`--sparse-window-tokens N` and `--sparse-refresh-ticks R` on `serve`, with
`TILERL_SPARSE_WINDOW_TOKENS` / `TILERL_SPARSE_REFRESH_TICKS` as the second
source and the module defaults (128 tokens = 8 pages, 8 ticks) as the fallback.
Omitted or empty everywhere, behaviour is unchanged.

They exist so a W1024/R32 deployment is a launcher edit rather than a source
edit. Both were module constants with several consumers, and the sweep that
needs them was patching four points from outside
(`scripts/probe_wr_sweep_worker.py`): `sparse_index.WINDOW_TOKENS`,
`sparse_index.WINDOW_PAGES`, `sparse_engine.WINDOW_PAGES` (a by-value import) and
`sparse_engine.SPARSE_REFRESH_TICKS`. A missed point is not a crash — it is a
pool, a tick width and a graph key at three different geometries.

## Why the override is applied once, at build

The window is read by the pool ledger (`memory.sparse_pool_num_blocks`), the
tick's own width (`sparse_engine.own_bound`, `sparse_runtime`), the scorers in
`sparse_index`, and the captured graph key. Applying it at `build_engine`, before
anything is sized, is what makes those one value. Moving the same call to after
the pool is sized is caught by the gate with a specific reading: **173 blocks
against the ledger's 397**.

## Speed — what is claimed

Nothing yet, and deliberately. This is a configuration surface, not a
performance change: at the defaults it runs the identical code path. The numbers
that matter are the sweep's, and they are the sweep's to publish.

| quantity | how | status |
|---|---|---|
| window × refresh sweep at W1024/R32 | `probe_wr_sweep_worker.py`, one arm per process | pending-remote |
| fidelity of each R arm vs same-W R=1 | TF logits, top1 agreement ≥ 0.99 | pending-remote |
| served tok/s at the chosen (W, R) | launcher flag only, no code change | pending-remote |

The memory cost of a larger window is expected to move the pool ledger linearly
in pages (`n_groups*k + WINDOW_PAGES + chunk_pages`), which is measurable from
the ledger on any card without the sweep; the sweep is for fidelity and rate.

## Gate

`tests/test_sparse_wr_flags.py`. A non-default value must reach **every**
consumer — the two modules, the refresh cadence, and the built engine's pool
against the ledger — and the same reading must be red without the flag. Four
mutations, each red on its own assertion: the window override removed, moved
after pool sizing (the 173/397 reading), the by-value `sparse_engine` mirror
removed, and the refresh write removed. The process-wide globals are restored by
an autouse fixture so a later test in the same process does not inherit a
1024-token window.
