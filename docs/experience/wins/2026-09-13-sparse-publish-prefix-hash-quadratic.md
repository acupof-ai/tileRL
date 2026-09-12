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

## H20 card-3 measured (27B, head 73183eb8, 6/12 GiB host/SSD)

| | #556 head | this fix |
|---|---:|---:|
| 256k sparse prefill | 996.3 s | **601.3 s** |
| cumulative spill write | 10.7 s / 1.1% | 11.3 s / 1.9% |
| spill effective rate | 1049 MiB/s | 1000 MiB/s |

395 s (~40%) recovered; the spill write is unchanged, confirming the saving is
the removed quadratic hashing and per-drop sync. 590 s remains (compute, D2H,
Quest bound scoring) and is profiled separately.

## Rule

A byte/second rate match is not an attribution: the "16 MiB/s spill" was the
16-token prefill chunk rate measured at the writer, not a disk cap. Time the
suspect call directly before optimizing it, and when a per-item cost scans a
growing prefix, check whether the key admits an incremental recurrence before
touching the I/O path.
