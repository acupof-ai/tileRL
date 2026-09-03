---
question: Why did widening the split-KV decode kernel to W query tokens make it emit NaN on one sequence length in ten?
source: H20 sm90, tilelang 0.1.13, tiny shapes (hkv=1, G=8, D=16) against _naive_paged
---

# A flash-decoding split starts at `t0`, and at W>1 its first tile can be entirely masked

`make_paged_attention_decode` gives split `sp` the tile range
`[t0, t1) = [sp*per, min(tiles, sp*per+per))` and runs the online softmax from
`k = 0` of *that* range. Every tile begins with

```python
T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
...
scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale * log2e)
```

At W=1 that was safe by accident. A non-empty split has `t0*64 < n`, and every
row's causal bound was `n-1`, so key `t0*64` was always inside it and
`scores_max` was finite after the first tile.

At W>1 row `i` is bounded by `hist + i%W` with `hist = n - W`, up to `W-1` keys
lower. When `n % 64` lands in `[1, W-1]`, the last tile starts above the low
chain positions' bound while its split is still non-empty. Both maxima are
`-inf`, `(-inf) - (-inf)` is NaN, `T.exp2(NaN)` is NaN, and `logsum` and
`acc_o` follow. The combine then computes `w[sp] = exp2(-inf - m) = 0` and
`l[0] += 0 * NaN` — **`0 * NaN` is NaN**, so the empty-slice contract that
handles a split with no tiles does not handle a split whose tiles are all masked.

## What it broke

Measured against `_naive_paged`, sweeping `n` at fixed `hkv=1, G=8`:

| n | W | tiles | per | KVSPLIT | NaN at chain positions |
|---:|---:|---:|---:|---:|---|
| 65 | 8 | 2 | 1 | 16 | 0-6 |
| 65 | 1 | 2 | 1 | 16 | none |
| 72 | 8 | 2 | 1 | 16 | none |
| 129 | 8 | 3 | 1 | 16 | 0-6 |
| 1025 | 8 | 17 | 1 | 64 | 0-6 |
| 65 | 2 | 2 | 1 | 16 | 0 |
| 66 | 2 | 2 | 1 | 16 | none |

Every `n % 64` in `[1, W-1]` whose last tile starts a split: at W=8 that is
224 of the first 4096 sequence lengths (5.5%) with `KVSPLIT=16` and 441 (10.8%)
with the 64-way split the long-context and under-filled-grid paths pick. It
does not thin out with context length; it doubles.

The sibling `make_paged_attention_mma` is immune, and for a reason worth
keeping: its `T.Pipelined` always starts at `k = 0`, whose mask
`0 + j < hist + bx*block_M + i + 1` admits `j = 0` for every row.

## Root cause

Generalizing a kernel along a dimension its guards did not mention. Nothing in
the split-KV epilogue said "a non-empty split holds at least one live key" — it
was a consequence of `hist = n - 1` that stopped holding when `hist` became
`n - W`.

## Fix

Guard the both--inf case so the split falls back to the empty-slice contract
the combine already handles — `PL = 0`, `PM = -inf`, `PO = 0`:

```python
scores_scale[i] = T.if_then_else(
    scores_max[i] == -T.infinity(accum_dtype), 1.0,
    T.exp2((scores_max_prev[i] - scores_max[i]) * scale * log2e))
acc_s[i, j] = T.if_then_else(
    scores_max[i] == -T.infinity(accum_dtype), 0.0,
    T.exp2(acc_s[i, j] * scale * log2e - scores_max[i] * scale * log2e))
```

`test_paged_attention_vs_naive` gains `(w, n) = (8, 65)`. Negative control: that
case fails against the unguarded kernel and passes with the guard, on 0.1.13.
The three rows that were already clean move by 0.00e+00.

## Rule

When a kernel gains a dimension, re-derive its *implicit* guards, not just its
indices. Here the arithmetic and the masks were all correct; what broke was an
invariant nobody had written down.

A parity case whose sequence lengths all fit one `block_N` tile cannot see any
split-KV geometry at all — only `sp=0` is ever non-empty. Sweeping widths while
holding `n < 64` sweeps nothing. Pick `n` so the split count, the tile count and
`n % block_N` each vary.

`0 * x` does not sterilize a NaN. An "unused" lane whose value is multiplied by
a zero weight still poisons the sum.
