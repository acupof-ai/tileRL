# 256k sparse prefill: per-drop prefix hashing was 40% of the wall-clock — 2026-09-13

## Context

After the #556 host-budget fix bounded the 256k OOM, the sparse 256k prefill
still took 997.6 s (3.8 ms/1k-token vs 0.84 at 128k). An initial rate match
blamed the synchronous mmap spill write, but a cumulative timer showed
`ColdSsdFile.write` was only 10.7 s (1.1%) of the 996.3 s prefill at
~1 GiB/s into page cache, with zero read-through, and the disk itself sustains
175–192 MiB/s even with fsync per page (see
[the #556 entry](2026-09-13-sparse-prefix-spill-host-rss-bounded.md)). A
60 s @200 Hz py-spy record in the post-budget phase named the real cost.

## What was slow — O(M²) prefix hashing + a device sync per drop

Top self frames in the post-budget profile:

| frame | self % |
|---|---:|
| `_page_hash` | 17.2 |
| `page_key` | 14.7 |
| `publish_dropped` tuple/genexpr | 11.4 |
| `note_boundary` | 5.2 |
| `has_bounds` | 5.2 |
| `_enforce_budget` | 1.6 |
| `ColdSsdFile.write` | 1.7 |

`SparsePrefixCache.publish_dropped` recomputed every page's content key with
`page_key(tokens, p)`, and `page_key` rehashes tokens `0..(p+1)·16` **from
zero** each call. Pages drop one at a time across the prefill, so at 256k the
~16k drops together executed `16·(1+2+…+16384)` python hash steps — quadratic
in context, binding only once the run is long (invisible at tiny and 128k). The
grow entry also detached/rechained by recomputing its chain hash. Separately,
`has_bounds` did `bool(bounds_valid[page])` on a CUDA bool tensor, forcing a
device sync on every dropped page.

## Fix (`src/tilerl/sparse_engine.py`)

- The prefix hash is rolling, so a publisher extends one running hash over each
  new page's 16 tokens and caches the per-page content keys and token tuple;
  the grow/frozen entry carries its chain hash and detaches from the stored
  bucket. Over P one-page drops `_page_hash` is now called exactly `16·P`
  times, not `16·Σ1..P`. Keys are bit-identical to the old `page_key`.
- A host `bytearray` mirrors `bounds_valid` (its only writers are `set_bounds`,
  `_grow`, attach/drop); `has_bounds` reads the mirror and never indexes the
  device tensor.

## Gates (CPU)

- `test_published_content_keys_are_bit_identical_to_page_key`: random prompts
  at 1/7/13 pages — every published key, the entry key list and chain bucket
  equal the from-zero `page_key`.
- `test_publish_hash_steps_stay_linear_not_quadratic_in_context`: exactly
  `16·128` `_page_hash` calls over 128 drops; fails with `16·Σ1..128` on the
  old head.
- `test_drop_reads_the_host_bounds_mask_without_touching_the_device_tensor`:
  the device mask is replaced with an object that raises on `__getitem__` and
  the per-drop `has_bounds` path stays correct; fails on the old head.
- full sparse + hermetic suites green.

## H20 card-3 measured (27B, 6/12 GiB host/SSD)

| head | 256k prefill | spill write |
|---|---:|---:|
| #556 f019a2e9 | 996.3 s | 10.7 s / 1.1% |
| first fix 73183eb8 | 601.3 s | 11.3 s / 1.9% |
| tail-hash + RAM counter c6f72be5 | **359.5 s** | 12.1 s / 3.4% |

996.3 → 359.5 s, **64% recovered** (3.8 → 1.37 ms/1k-token; the dense 128k row
is 0.84). The first fix removed the quadratic rehash and the device sync; a
second profile then showed `_ensure_prefix` still rebuilt the whole token
tuple per drop and `_shared_ram_bytes` summed every shared record per budget
iteration — converting only the grow-only tail and tracking shared RAM as a
running counter removed those. The spill write is flat at ~10–12 s / ~1 GiB/s
across all three heads, so none of the saving is I/O. The residual 347 s is
fp4 forward/MLP compute in the profile (`linear_fp4` 14% + `forward` 17% +
`_mlp` 19%) — the model's real prefill cost, not host bookkeeping.

## Rule

A byte/second rate match is not an attribution: the "16 MiB/s spill" was the
16-token prefill chunk rate measured at the writer, not a disk cap. Time the
suspect call directly before optimizing it, and when a per-item cost scans a
growing prefix, check whether the key admits an incremental recurrence before
touching the I/O path.
