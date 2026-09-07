# The CPU twin of the sm70 prefill cell, and two TileLang limits it hit — 2026-09-07

> Step 2 of the accepted sm70 prefill cell. Tile shape from step 1
> (`wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md`); the measurement
> that justified the work is
> `wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md`.
> No sm70 code — routing is step 3.

## What landed

`make_paged_attention_prefill` in `kernels.py`, registered on cpu. Same contract
as `make_paged_attention`; what differs is the shape of the work. The query tile
is in the grid — `T.Kernel(ceildiv(S, block_M), H, B)` — so one K/V tile serves
`block_M` rows. `paged_attention_split` opens `Qf[d] = Q[bb, tt, hh, d]`, one
query row per block, which is right for a verify width ≤ 8 and re-reads K/V once
per row for a 512-row prefill chunk.

Defaults `block_M=64`, `block_N=16`: step 1's shared-memory result. `kv_dtype`
is a maker parameter rather than f32 throughout, so the fp16-tile variant is one
instantiation and not a second kernel.

## The two cuts a query tile adds

A one-row kernel gets both for free, which is why they are where the bugs are.

The tile's K/V range is bounded by its **last** row (`upper = hist + last + 1`),
so every row also masks **its own** causal cut inside the tile:

```python
sc[i, j] = T.if_then_else(
    (p < upper) and (p <= hist + bx * block_M + i),
    dot[0] * scale, -1.0e30,
)
```

And `S` need not be a multiple of `block_M`, so the Q load and the output store
both guard `t < S`.

## Two TileLang limits, both hit on the first compile

**`T.reduce_max` / `T.reduce_sum` have no CPU implementation.** `tl.reduce`
resolves per target and `"c"` is not registered:

```
InternalError: Check failed: (matched_impl != nullptr) is false: tl.reduce
requires a target-specific implementation, but no reduce implementation is
registered for {"kind":"c","tag":"","keys":["cpu"]}
```

The natural writing of a tiled online softmax is `T.reduce_max(sc, m, dim=1)`
over the score tile, and it compiles on sm70/sm90 and not here. The twin uses
the serial-scalar idiom `make_paged_attention` already uses; the sm70 cell will
use the intrinsics, and that divergence is commented at the loop so the two do
not silently drift.

**A read-modify-write of a 2D fragment inside a 1D `T.Parallel` is rejected.**

```
InternalError: Check failed: (StructuralEqual()(it->second.indices, indices))
is false: sc: (i, j) and (i, j)   -->  kernels.py:806
```

`for i in T.Parallel(block_M): ... sc[i, j] = T.exp(sc[i, j] - mn[0])` — reading
and writing `sc[i, j]` under a parallel `i` with a serial `j`. The module
docstring already warns that a serial `j` inside a parallel `i` **miscompiles on
Metal**; on the CPU target it is a hard error instead, which is the better
failure. The rescale sits in `T.serial(block_M)` with `T.Parallel(D)` inside.

Both are properties of the CPU target, so neither would have surfaced on the
V100 — and the second would have surfaced on Metal as wrong numbers rather than
a compile error.

## The gate

Eight arms in `test_ops_parity.py` against the existing `_naive_paged`, at
`allclose(rtol=1e-2, atol=1e-2)`, max observed error **6.6e-07**:

| arm | S | hist | tile |
|---|---|---|---|
| prefix 0, one tile | 16 | 0 | 16×8 |
| prefix 0, several tiles | 48 | 0 | 16×8 |
| with history | 48 | 96 | 16×8 |
| S == block_M | 16 | 33 | 16×8 |
| S == block_M + 1 (tile straddle) | 17 | 33 | 16×8 |
| ragged S and hist | 23 | 47 | 16×8 |
| block_N > block_size | 24 | 40 | 16×32 |
| uneven GQA (H=6, Hkv=3) | 20 | 44 | 16×8 |

Parity is against the naive reference rather than `paged_attention_split`
directly: split returns PO/PM/PL partials that need
`paged_attention_split_combine` to become an output, so comparing against it
asserts on a composition of two kernels — and `test_paged_attention_vs_naive`
already gates split against the same reference, so the guarantee is transitive
with one fewer moving part.

## Eight passes proved nothing until the harness went red

A kernel I had just written passing every arm is the case where the arms are
wrong. Six mutations, each in a **fresh interpreter** — a reload leaves both the
`.pyc` and the TileLang JIT cache in place, and a cached kernel reads as a
passing mutation:

| mutation | result |
|---|---|
| *(unmutated)* | **PASS** 3.576e-07 |
| drop the per-row causal cut | FAIL 8.275e-01 |
| row cut uses the tile's last row | FAIL 8.275e-01 |
| off-by-one in the row cut | FAIL 2.496e-01 |
| drop the acc rescale | FAIL 1.972e+00 |
| drop the `l` rescale | FAIL 9.985e-01 |
| Q row ignores the tile offset | FAIL 7.286e-01 |

The first two are the same magnitude because both make a row attend past its own
bound; they differ in which rows, and the arms catch each.

## The doc table is a fourth file

Registering one CPU kernel moved `_CPU_KERNELS` from 15 entries to 16, and every
accelerated cell inherits the floor, so `test_support_matrix` went red on
**three** arch rows at once:

```
metal is (16, 3, 0, 13) but the doc's row reads (15, 3, 0, 12)
sm90  is (42, 9, 26, 7) but the doc's row reads (41, 9, 26, 6)
sm70  is (24, 2, 8, 14)  but the doc's row reads (23, 2, 8, 13)
```

Adding one kernel to the CPU cell is a four-file change — `kernels.py`,
`registry.py`, the test, and `docs/support-matrix.md` — and the fourth is the
one that is easy to miss. That gate exists because every number in that table
was wrong on 2026-09-05 with nothing checking it; it did its job here.

## What is not gated yet

**No end-to-end test, deliberately.** Nothing routes to this cell —
`backend.py:933` still sends every sm70 attention call to
`paged_attention_split`, and the dispatch is step 3. An end-to-end run today
would exercise the split kernel and pass whatever the twin does: a green that
proves nothing. It lands with the routing.

## Rule

A portability limit found on the target you develop on is cheaper than the same
limit found on the target you ship to. Both limits here are CPU-target
properties that the sm70 cell will not have, and one of them
(`T.reduce_*` per-target registration) means the twin and the cell cannot share
that line — worth a comment at the divergence, because a twin that quietly
stops mirroring is a parity gate testing itself.
