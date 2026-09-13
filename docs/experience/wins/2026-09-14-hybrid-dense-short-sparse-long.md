# Hybrid serve: short prompts dense on the captured graph, long prompts sparse — V100, 2026-09-14

> Status: pending-remote. CPU gates green (`tests/test_sparse_engine.py -k hybrid`);
> the device number is ops-0b's run on `proto/sparse-min-tokens`.

## Context

Sparse k=128 is accuracy-equivalent but slower on short prompts; the dense
captured graph is fast for them but cannot hold a long context. The hybrid
engine (`--sparse-min-tokens N`) fixes the mode per request at submit: prompts
up to N tokens run dense on the precaptured decode graph and pin their whole
context in the device KV pool (no sparse sharing); longer ones run sparse.
Ticks never mix the two BatchKv geometries — while both modes have running
rows, whole ticks alternate per mode (round-robin). The dense graph is
precaptured before traffic and sparse ticks run **eager** deliberately: eager
sparse is token-exact on sm70, so hybrid does not depend on the sparse-graph
capture or its warmup-frame fix (#585).

## Device checks (ops-0b, V100 n37-002-027)

Boot with `--sparse-k 128 --sparse-min-tokens 8192 --draft <mtp> --depth 1
--decode-graph`.

1. **Dense precapture at boot:** the server log shows the dense graph
   precapture completing before the first request (`precapture()` over the
   (B,W) grid); `sparse_graph_on` stays false.
2. **Short request speed:** one prompt under 8192 tokens, think on,
   `≥ 50 tok/s` decode.
3. **Concurrent modes:** a 32k sparse request answers correctly while a short
   dense request streams; read `/health` `dense_mode_ticks` and
   `sparse_mode_ticks` (both must rise; no tick carries both modes — asserted
   on CPU), and record the short request's tok/s under alternation.
4. **MMLU smoke:** the served default MMLU smoke stays green with the hybrid
   flags.

## Rule

A hybrid of a fast bounded path and a slow unbounded one needs the mode fixed
before admission (a row cannot re-pin mid-request) and a capacity route for
prompts the bounded path cannot hold — the dense pool cannot quietly become a
permanent head-of-line block.

## Results

| date | commit | machine | config | precapture | short tok/s | 32k correct | short tok/s under alternation | MMLU |
|---|---|---|---|---|---:|---|---:|---|
| 2026-09-14 | pending | n37-002-027 V100 | N=8192 k=128 d1 graph | | | | | |

Raw artifacts: `<server log>`.
