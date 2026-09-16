# Cold-fill prefill chunk cap (192 vs 512)

When to run the sparse engine with the larger prefill chunk, and why the shared
server keeps the smaller default.

## The knob

One sparse prefill tick has a token budget

```
min(max_num_batched_tokens - n_decode_rows, sparse_prefill_cap)
```

- `max_num_batched_tokens` defaults to **512** (`--max-batched-tokens`).
- `sparse_prefill_cap` (set in the engine from `--sparse-prefill-tokens`):
  - in **hybrid** mode (any `--sparse-min-tokens` N) the cap defaults to **192**;
  - in **pure sparse** mode (`--sparse-min-tokens` omitted → 0) the cap is
    **0, i.e. uncapped**, so a sparse prefill tick takes the full **512**.
- The cap is applied only on the hybrid sparse-mode branch
  (`if mode_sparse and self._sparse_prefill_cap`). It never bounds a dense row
  or a decode tick.

192 is about one second of V100 sparse prefill at ~191 tok/s; 512 is the tick
budget both servers already size their pool for. The sparse pool size reads only
`max_num_batched_tokens`, so moving the cap 192→512 needs no pool/block change.

## Use 512: a dedicated / offline long-context cold fill

For a job that owns its slot and wants the shortest cold long-context fill,
**no code change is needed** — either of these gives 512-token sparse chunks:

1. **Pure sparse (recommended for offline cold fill)** — drop
   `--sparse-min-tokens` from the hybrid command. With it absent the cap is 0 and
   each sparse prefill tick runs at the full 512.
2. **Hybrid with a raised cap** — keep `--sparse-min-tokens` (short prompts stay
   dense) and add `--sparse-prefill-tokens 512`; long sparse fills chunk at 512
   instead of 192.

Leave the rest as the host supervisor sets it: `--sparse-k 128`, `--depth 1`,
`--decode-graph`, `--max-batch` (lower for a single-user endpoint),
`--max-ctx`, and the cold host/SSD tier. See
[serve-v100.md](serve-v100.md) and [serve-h20.md](serve-h20.md) for the host
launchers.

Estimated effect, **an extrapolation awaiting a device measurement**: a 128 k
cold fill of ~1037 s at 192 comes down to roughly **770-900 s** at 512.

## Keep 192: a shared hybrid server serving concurrent short requests

Do not raise the default. 192 bounds how long a dense row waits while a sparse
fill owns a tick (~1 s). At 512 a short request that lands during a long cold
fill waits up to ~3 s per tick — a shared, concurrent deployment becomes
fill-bound on the short request. The larger cap is for a slot a single offline
job owns, not the shared server.

## Scope: first-token latency only

- This changes **cold-fill wall time / time-to-first-token**: fewer, larger
  prefill ticks.
- It does **not** change steady decode tok/s — no decode path, KV layout,
  attention kernel, or page selection changes.
- It is not the speculative-decode W-window (draft verify width) line; keep the
  two separate when reporting.

## Verifying on a device window

Run one slot, one 128 k cold-sparse prompt, cap 192 vs 512, and compare
time-to-first-token and total fill wall. Ensure no concurrent short request is
in the run (or time its wait separately) so an offline fill number is not read
as a serving improvement.
