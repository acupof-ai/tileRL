# DFlash2 drafts through the engine — H20 GPU 7, 2026-09-03

> Status: **wired and correct; negative speedup as wired.** 6.18-of-8 accepted,
> 5.5x fewer trunk forwards, 1.67x slower wall clock. The drafter is 68.4% of
> the tick and runs one row at a time in Python. Not a default flip.

## Context

`DFlash2Head` had no path through the product. The engine reaches a draft head
through `forward()` and `confidence()`; this head has neither, and `_draft_step`
is autoregressive besides — one token per iteration, each fed back — while the
head emits a whole block in one pass. `draft()` had two callers on `main`, both
scripts. The 5.80-of-8 on record is a probe measurement on a path the engine
cannot take, which
[the corrected entry](2026-09-03-dflash2-block-drafter.md) now says at the top.

This entry is the engine wiring and the numbers measured on the tick. The two
acceptance figures are kept apart on purpose: **5.80 of 8 is the probe, 6.18 of 8 is
the engine.** They are different code paths.

## What Worked

### The branch is one method; the verify half is untouched

`_draft_step` dispatches on the head. `_verify` did not change: a block chain is
`[anchor, *drafts]` exactly like a chain head's, and accept-the-leading-run does
not care how the drafts were produced. Nor did `submit`/`poll`, `StepLimits`, or
one-forward-per-tick — a block drafter *removes* forwards (one block pass in
place of `spec_depth` sequential draft passes), it does not add any.

### `aux_layers` was defined and never passed

`Model.forward` has always taken `aux_layers`, and its docstring names the
DFlash2 drafter. Nothing in `src/` passed it, so `hidden_out` held one entry —
the pre-final-norm state the NextN head wants — and the block head's `fc` never
saw the five taps it consumes (`target_layer_ids` = 5, 19, 33, 47, 61 on a
64-layer trunk). `_run_forward` and the captured decode graph now both pass it;
`()` for a chain head, so that path is bit-identical.

### The object with a lifetime is the projected K/V, not the aux hidden

`context_kv` is per-position pure — `fc → hidden_norm → per-layer
k_proj/k_norm/rope, v_proj`, no mixing across positions — so a context position
is projected the tick it commits and never again. Keeping the aux hidden and
re-projecting instead costs, at T=512 B=8:

| | keep the aux, re-project per tick | keep the projected K/V |
|---|---:|---:|
| per-tick FLOP | 1.5 TFLOP | 23.5 GFLOP |
| per-tick ms (est. at ~150 TFLOPS effective) | ~10 | ~0.16 |
| per-token bytes | 50 KiB (25600 × bf16) | 40 KiB (5 × 2 × 8 × 128 × f32) |

10 ms sits on a 47.2 ms W=8 B=8 verify tick — **+21%**, against the 1.79x-of-a-
W=1-tick that [the width-W entry](2026-09-03-width-w-verify-tick-on-the-decode-path.md)
bought. Re-projecting would have spent the acceptance win before it was
measured, which is why the cache is a design decision and not an optimization.
It is also the shorter code: an incremental `context_kv` over the newly
committed positions plus a `cat`. Trimmed to the head's `sliding_window` (2048),
past which every block slot masks the context anyway.

### The width is the checkpoint's block

`block_size` is 8, so a draft returns 7 and the verify width is 8 — exactly
`_MAX_VERIFY_W`, with no headroom. `DFlash2Head.spec_depth` is `block_size - 1`
and `build_engine`/`Engine` read it; passing `spec_depth` alongside a block
drafter **raises** rather than being ignored, and a `block_size` past the verify
tile raises with the consequence named (the tick would leave the decode path for
the M-tiled prefill kernel). `verify_lens`/`survival` are switched off on this
path, not merely unused: the head has no per-slot confidence, so the block is
kept whole.

### Three failures that were silent, closed at the source

- `_quantize_draft` packs any `[N,K]` with both dims ≥ 128. That is the two
  selector codebooks, `[248320, 256]` tables the walk indexes by token id.
  Shape cannot tell a codebook from a projection, so the head names them
  (`no_quant`), the way `read_head_params` already names `embed_tokens`.
- The head read bare `self.params[key]` through `backend.linear`, so the first
  re-serve would `KeyError` on `fc`. It now reads through `Model._linear` and
  gets the same `.w8/.wscale` dispatch the trunk has.
- Prefix reuse is incompatible and raises with the reason in the message. The
  context K/V is built only from positions this process forwarded; an adopted
  prefix skips them, and the drafter would attend over whatever the recycled
  blocks hold — a failure that degrades acceptance and looks like a weak
  drafter rather than a bug.

### fp8 costs acceptance, and the threshold was set before the number

The engine re-serves a draft head fp8. The control, on the probe path so it is
comparable to the 5.80 on record, GPU 7, tilelang 0.1.13:

| head served as | mean acceptance of 8 | per prompt |
|---|---:|---|
| bf16 (the record) | 5.80 | 4, 8, 7, 6, 4 |
| fp8, codebooks excluded | 5.20 | 4, 8, 7, 3, 4 |
| fp8, codebooks and `fc` excluded | 5.80 | 4, 8, 7, 6, 4 |

The decision rule was written down first: ship fp8 everywhere at ≥ 5.0, because
the B=1 break-even is 2.41 accepted tokens and excluding `fc` costs a bf16
projection on the generic path, measured at 9.7 ms against 0.13 ms fp8. 5.20
clears it, so `fc` stays quantized. The honest read of the gap is narrow: it is
one prompt of five moving 6 → 3, and the probe cannot resolve below ~0.6 at
n=5. `fc` moving into `no_quant` is one string if the engine-tick number
disagrees; at n=200 on the engine it does not — 6.18 of 8 is above the 5.80
probe, so fp8 on `fc` costs nothing measurable on the path that ships.

## The gate, and the mutations it catches

`tests/test_dflash2.py`, CPU cell, tiny trunk + a random block head. Output
equality alone proves nothing here — a rejected draft costs a trunk row and
never a token — so the gate compares the block the engine drafted against
`draft()` recomputed from the whole token prefix, across two prefill chunks.

| mutation | caught by |
|---|---|
| `_draft_step` dispatch removed | `AttributeError` |
| `aux_layers` dropped from `_run_forward` | shape mismatch |
| anchor off by one | content gate, and the anchor assert |
| context cache dropped | content gate |
| sliding window shortened to 1 | content gate |
| block start one position late | content gate |
| prefill-phase boundary reverted | the hole check, by name |

The first gate written passed under two of these; the stub that made a whole
block commit also ignored the anchor and the hidden, so everything upstream of
it was unobservable:
[errors/2026-09-03-an-oracle-stub-blinds-the-gate-it-sits-in.md](../errors/2026-09-03-an-oracle-stub-blinds-the-gate-it-sits-in.md).

## Results

H20 GPU 7, tilelang 0.1.13, Qwen3.8-27B NVFP4 trunk, greedy, thinking off,
`max_new_tokens=512`, 200 GSM8K test questions, B=8, decode graph on, prefix
store off. Both arms in one process, same session.

| | wall | tok/s | tok/decode-fwd | block accepted | GSM8K |
|---|---:|---:|---:|---:|---:|
| base | 278.6s | 232.3 | 7.80 | — | 170/200 = 85.0% |
| spec-w8 | 466.5s | 139.4 | 42.99 | **6.18 of 8** | 167/200 = 83.5% |

**The drafter works and speculation loses anyway.** 6.18 of 8 accepted is 77%,
above the 5.80 probe figure, and it buys 5.5x fewer trunk forwards — and the
wall clock is 1.67x worse. A spec tick costs about 9.2x a base tick.

1322 wide ticks, every one at width 8; 0 NaN trunk logits. 152 of 200
completions differ from the base arm's, which W>1 has never promised not to do
([w8-not-lossless](2026-09-03-w8-verify-tick-is-not-lossless-on-the-27b.md)). The 1.5-point accuracy gap
is 3 questions against a 2.5-point binomial sd at n=200 — it does not support a
regression claim either way.

### Where the tick goes

`py-spy record`, 40s of the spec arm at 100 Hz, 3998 samples, attributed to the
outermost engine phase in each stack so the buckets do not overlap:

| | share |
|---|---:|
| drafter (`_draft_block`) | **68.4%** |
| — of which `block_hidden` | 52.9% |
| — of which `context_kv` | 8.2% |
| verify + sample | 18.4% |
| trunk tick (graph replay) | 13.2% |

The largest single self-time frame is not in the product: `traced_verify` in
the benchmark script, **13.8% of the arm**, from `bool(bad[i])` per row inside
the NaN tracer — eight GPU syncs a tick on the spec arm and none on the base
arm, because the base arm never enters `_verify`. The comparison was biased
against speculation by its own instrument. Fixed to one `bad.any()` sync; the
throughput rows above still carry it, so they are the pessimistic reading.

Estimates from those shares, not measurements: tracer removed, 402s (1.44x
slower); tracer removed and the drafter batched, ~83s (3.4x faster than base).
The second number is what the batching work is worth, and it is why this is a
`ponytail` marker and not a default flip.

The trunk-tick half of that arithmetic is measured, not extrapolated: a B=8
W=8 graph replay is 47.205 ms against the same batch's W=1 at 26.361 ms,
**1.79x**, with W=1 reproducing to 0.8% as its own control and 1.79 on GPU 7
against 1.81 on GPU 5
([width-w entry](2026-09-03-width-w-verify-tick-on-the-decode-path.md)).
1.79x over 5.5x fewer forwards predicts the spec arm's trunk time at 0.325 of
the base arm's; the profile puts it at 61.6 s against a 278.6 s base that is
mostly trunk and sampling, so 0.22 — two routes agreeing to the order, with
the gap in the direction the base arm's non-trunk work explains.

**Two limits on the 3.4x, both from review, neither closed here.** First,
Amdahl on a self-time share is not a speedup: removing 68.4% predicts 3.16x
only if the remaining 31.6% is unchanged, and a batched drafter is not zero —
it is a smaller unmeasured number, so the honest form is "≤3.4x minus the
batched drafter's own cost," and that subtrahend is the whole question.
Second, **6.18 of 8 was measured with the drafter running per row, and it is
an input to the 3.4x.** `path`'s walk carries a data dependency on `prev`, so
batching it is either B independent walks or a restructure, and a restructure
can move acceptance. An estimate that holds acceptance fixed while changing
the code that produces it is assuming its own answer.

What settles it is not a profile: a synthetic verify-only arm — trunk at W=8,
B=8, graph on, drafts replayed from a trace recorded at the acceptance a real
run achieved, no drafter in the process. That prices the ceiling with none of
the drafter in it. Pending.

### Why the drafter costs what it does

`_draft_block` walks the rows one at a time. Per tick at B=8 that is 8 serial
`block_hidden` forwards — every head layer over the 8 block slots against a
2048-position context — and `path` takes an `int(score.argmax())` per slot,
so 7 GPU syncs per row, 56 per tick. None of it is inside the decode graph.

Both halves batch. Every helper (`_lin`, `_heads`, `_rope`, `_conv_in`,
`_conv`) already indexes `h.shape[:2]`, and `_attend` already carries a batch
dimension; the only per-row state is the context length, the context positions
and the block's start position. Padding the contexts to a common length and
filling the pad slots' `k_pos` with a large negative value makes the existing
`q_pos - k_pos >= window` mask cover them, with no second mask. `path`'s walk
is sequential in the slot and parallel in the row: 7 syncs a tick, not 56.

### One number this entry does not explain

Peak CUDA 25.86 → 38.62 GiB, +12.76 GiB for the draft head. The fp8 head is
about 1.9 GB and the context K/V at the 40 KiB/token measured above is 0.65 GB
at 8 rows x 2048. That leaves roughly 10 GiB unaccounted for. Unresolved.

## Rule

Acceptance is not throughput. 6.18 of 8 and 5.5x fewer trunk forwards sat
beside a 1.67x slowdown, because the drafter that produced the acceptance was
not on the same execution path as the trunk it was saving work for — Python,
per row, outside the graph. Measure the tick, not the accept rate.

Profile the instrument before quoting the number it produced: the NaN tracer
was 13.8% of the arm it was measuring, and it charged only one of the two arms.
