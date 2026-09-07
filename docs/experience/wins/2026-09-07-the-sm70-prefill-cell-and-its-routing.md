# The sm70 prefill cell, and an A/B arm that compared a kernel to itself — 2026-09-07

> Step 3 of the accepted sm70 prefill cell. Measurement:
> `wins/2026-09-07-the-sm70-prefill-n2-is-one-kernel-at-30x-its-floor.md`.
> Tile: `wins/2026-09-07-the-sm70-prefill-tile-is-64x16-f32.md`.
> CPU twin: `wins/2026-09-07-the-cpu-twin-of-the-sm70-prefill-cell.md`.
>
> **Status: pending-remote.** No V100 in CI and none on this machine, so the
> three sm70 arms skip here and the acceptance numbers do not exist yet. The
> window writes the accept-or-reject line; there is no CHANGELOG entry until it
> runs.

## What landed

`make_paged_attention_prefill_sm70` in `kernels_attn.py`, and the routing that
reaches it. sm90's structure with none of its instructions — Volta has no bf16
and `T.gemm` lowers to fp16-only `mma.sync.m8n8k4` — so QK<sup>T</sup> and PV are
hand-tiled `T.Parallel` MACs over shared Q/K/V tiles with f32 accumulators, and
the online-softmax rescaling is the sm90 cell's arithmetic verbatim.
`T.reduce_max`/`_sum` are used here where the CPU twin could not (they are
registered per target, and `"c"` is not one). `KVSPLIT` is gone: 8 query tiles ×
24 heads is 192 blocks against 80 SMs, so the card fills without splitting the
history.

`kv_dtype` is a maker parameter, so the fp16 rung is an instantiation rather
than a second kernel, and `--prefill-kv-dtype f32|f16` reaches it from the
profiler without editing the serving tree.

## One predicate, and a gate that counts call sites

```python
def is_prefill_width(s: int) -> bool:
    return s > _MAX_VERIFY_W
```

Defined beside the constant, called once. The gate asserts the **call-site count
by AST**, not the text: a grep for the expression passes while the copy computes
something else, which is a failure this seam has had before.

A second check asserts that no *other* comparison of a query width against
`_MAX_VERIFY_W` exists outside the predicate's body. Writing it surfaced two
hits, and only one is a re-derivation risk:

- `assert _MAX_VERIFY_W < _WY_CHUNK` — two constants, no query width. Excluded
  by requiring the comparison to mention `s`.
- `chain = s <= _MAX_VERIFY_W and ...` — the sm90 decode tile, asking whether
  the GQA group fits the M tile. A different question about the same constant,
  on an arch this cell does not serve. Allowlisted **structurally** (it must be
  the node inside the `chain` assignment), not by count, so a new one on the
  sm70 path fails.

Four mutations, four reds, each on a different assertion: predicate inverted,
off-by-one, dispatch re-deriving instead of calling, and a second call site
added.

## The A/B arm compared a kernel to itself

The same-completions gate runs one prompt with `is_prefill_width` true, then
monkeypatches it false so the same chunk goes down `paged_attention_split`. On
cpu, with the sm70 skip lifted, all four tests passed. Then three mutations of
the twin — dropped per-row causal cut, an off-by-one in it, dropped acc rescale
— left **all four still passing**.

On cpu the dispatch never enters the `arch == "sm70"` branch, so both sides of
the monkeypatch resolve to the same `paged_attention`. The arm compared a kernel
against itself and agreed, as it must.

The fix is for each arm to assert *which kernel ran*, by spying on
`backend._kernel` and requiring the expected name in the call log. With it, the
same cpu run reports **3 failed, 1 passed** — the vacuous arms now say so. On
Volta they run for real.

A monkeypatch on a branch predicate is inert on any target that does not reach
the branch, and the failure is green.

Which kernel each arm actually hits, per target:

| arm | on sm70 | on cpu |
|---|---|---|
| same completions, predicate true | `paged_attention_prefill` (this cell) | skipped |
| same completions, predicate patched false | `paged_attention_split` | skipped |
| T=1 and T=8 no-regression | `paged_attention_split` | `paged_attention` |

The cpu column is why the arms skip rather than run everywhere:
`paged_attention_split` is registered on sm70 only and is absent from
`_CPU_KERNELS`, so off-Volta the "split" arms cannot reach split at all.

## An all-masked row produces NaN

A query tile's last rows can have every key masked — the tile's K/V range is the
last row's, so an early row in a later tile sees `-inf` across a whole `block_N`
strip. Then `m[i]` stays `-inf` and the rescale computes
`exp(-inf - -inf)` = NaN, which propagates through `logsum` into every output
element of that row.

Guarded at both sites:

```python
mscale[i] = T.if_then_else(m[i] == -T.infinity(accum), 0.0, T.exp(mprev[i] - m[i]))
sc[i, j]  = T.if_then_else(m[i] == -T.infinity(accum), 0.0, T.exp(sc[i, j] - m[i]))
```

`paged_attention_split_combine` carries the same guard for the same reason
(`kernels.py`, "an all-empty row would make this exp2(-inf - -inf) = NaN"), and
the CPU twin does not need it because its `m` initialises to `-1.0e30` rather
than `-inf`, so the subtraction is finite. That difference between twin and cell
is exactly the kind a parity gate cannot see when the twin is the reference.

## The registry moves the doc table again

sm70 now **overrides** `paged_attention_prefill` rather than inheriting it, so
the support-matrix row goes 2 → 3 overrides and 14 → 13 same-as-cpu. Third
registry change in this sequence, third time the gate caught the doc.

## What the window measures

`scripts/prof_prefill_ops.py --model qwen38-27b --tokens 2048,8192,16384,21727
--selfcheck`, three gates from the approach note, all required:

1. the 16,384 arm's per-chunk slope falls by at least **6x** from 75.98x;
2. the ratio to the compute floor moves down from **29.9x** toward it — still
   above ~10x means the tile landed and the schedule did not;
3. TTFT at 16,384 falls. Reject if it does not, whatever the microbenchmark says.

**Both rungs run at 16,384 regardless of the f32 result**, and the 2,048 arm is
dropped to pay for it. Running fp16 only when f32 looks bad answers the wrong
branch: fp16 doubles residency without changing the tile, so it is what
separates "the tile did not help" from "fp32 arithmetic is the wall" from "the
occupancy model is wrong and it spills" — and if f32 clears the gate, skipping
fp16 leaves the size of the remaining gap unknown, which is the number that
decides whether the GQA step or the tensor-core rung comes next. The 2,048 arm
is 8.5% attention and settles nothing.

**Gate 3 binds.** A cell that improves the slope 8x and leaves TTFT flat is a
reject, not an accept with a caveat: it would mean the time is somewhere this
analysis did not look, and the microbenchmark would be measuring a cost that is
not on the critical path.

Which of `T.Pipelined` / serial lowered is printed in the arm line.

## Rule

Ask what a green from this arm would prove *on the machine that will run it*. An
A/B test is only a test where the switch changes what executes; everywhere else
it is two runs of one code path, and it passes hardest exactly when the code is
broken. The cheap fix is to assert the mechanism ran.
