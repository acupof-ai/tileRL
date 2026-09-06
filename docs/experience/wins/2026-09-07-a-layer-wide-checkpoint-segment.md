# A layer-wide checkpoint segment — H20 sm90, 2026-09-07

> Status: **Accepted as the shape-picked arm.** `"mlp"` stays the default; `"layer"`
> is selected at `train.py:166` when T exceeds the measured bracket's low end.

## Verdict

| row | `segment="mlp"` | `segment="layer"` | |
|---|---:|---:|---|
| gen 1024 `backward_secs`, paired | 69.748 s | 75.259 s | **1.079x — costs 5.51 s** |
| gen 4096 forward peak | 54.038 GiB | 14.896 GiB | **3.63x smaller** |
| gen 4096, does the step fit | no (OOM, 290.00 MiB) | **yes, 518.5 s** | **the shape that could not run, runs** |

The two point opposite ways, so neither arm is deleted: the layer segment buys a shape
that did not exist before and costs 1.079x on backward where the MLP one already works.
`_MLP_SEGMENT_MAX_T = 1280` in `train.py` switches at the low end of the measured bracket,
and the ponytail line there names both endpoints so a later measurement can move it.

## Context

`autograd.checkpoint` wrapped `_mlp_body` alone, so every layer's attention and
GDN activations stayed on the tape for the whole forward:
[the MLP-only measurement](2026-09-06-checkpointing-covers-the-mlp-only.md) put
**697.8 MiB retained per segment against the 85.0 MiB the wrapper stores**, i.e.
the checkpoint explained 12% of a 42.929 GiB accumulation. This entry is the
obvious follow-up — make the segment the whole layer — and its point is whether
the obvious thing is worth defaulting to.

`forward(..., segment=)` takes `"mlp"` or `"layer"`, and `_step` selects between them by T
at `train.py:166`.

**How the arms were selected while measuring, before that selector existed.** The two
forward calls that build a tape are `train.py:166` (the shipped training path, via
`_step` → `run`) and `scripts/prof_forward_memory.py:138`; nothing else in `src/`,
`scripts/` or `packages/` passes `segment=`. Each arm was therefore produced by editing
that one line in the measured tree — rows 1 and 3 in `train.py`, row 2 in the probe — not
by a flag threaded through `rl_step`/`grpo_loop`/`cli`. Deliberate: the same line is where
the shipped selector now lives, so the measured arm was the shipped path with one word
changed, and a plumbing chain added for the measurement would have been deleted by the PR
that resolved it.

This mattered more than a convenience. When I wrote it down I found that **`segment=` was
reachable from nothing but the tests** — I would have arrived on the card with an arm I could
not select. Both sed round-trips were validated byte-exact locally before the pod ran them,
because a failed revert would have made the second arm a silent duplicate of the first.

## Why a layer can be a segment at all

A checkpoint segment must be pure: the replay runs the same callable a second
time, so anything it reads that its own forward wrote comes back wrong.

- **Full attention is pure under training**, via two separate gates rather than
  the one I first wrote. `write_tokens` at `model.py:341` sits in the `else` of
  `if kv.dense` (:322), and the fused sm90 `attn_prep` — which also writes KV — is
  gated on `not kv.dense` at :282, so under training `qn` is None and that path is
  not entered either. Both KV-write routes are off; the training path is the
  tensor path.
- **GDN is the only impure layer.** `state_gather` at `model.py:411` reads the
  recurrent state that `state_scatter` at :427 advances, so a replay that
  re-gathers reads its own forward's output as its input.

Line numbers are on this branch (`240dd67`); on `origin/main` the same three are
407 / 420 / 434.

The fix is asymmetric, and not the one in the approach note: **the gather moves
out of the segment, the scatter stays in.** `forward` gathers state and conv
window before entering the checkpoint and hands them to `_gdn` as
`state_in`/`window_in`; the scatter stays inside because it is not a taped op and
recomputes the same values from the same handed-in inputs, which makes the replay
idempotent rather than merely harmless.

**A tuple-returning segment is not available.** `Tape.record` takes a single
`output` tensor and `backward` shape-checks `grad_output` against it, so a segment
cannot hand its state back out. That constraint is what forces the asymmetry above
rather than the tidier "move both boundary ops outside".

`win_parity` is asserted unchanged across a segmented forward: a flip mid-forward
would send a replayed scatter to the other plane. Training never flips it (the
flip is in `_gdn`'s serving branch), so this is an assertion, not a fix.

**The first version of that assertion was broken two ways at once**, and the two
hid each other. `parity0` held a reference to the live pool tensor, so
`win_parity == parity0` compared it against itself and could never fail; and
`win_parity` is `[num_slots] int32` (`kv_cache.py:229`), so above one slot the
comparison raises `Boolean value of Tensor with more than one value is ambiguous`
rather than asserting. `_training_kv` sizes `num_slots` by batch (`train.py:47`)
and the test helper used batch 1 — the one shape where a 1-element tensor
bool-ables and both defects are invisible. Found by selecting the arm at
`train.py:166` and running the real `grpo_loop`: **11 tests in `test_rl.py` went
red.** Now `.clone()` + `torch.equal`, with a batch-2 arm.

## What the CPU half establishes

Tiny model, CPU target, `segment="layer"` against `segment="mlp"`:

| check | reading |
|---|---|
| tape ops left outside the segments | **33 → 5** — everything but embedding, final norm and lm_head moves inside |
| named parameter gradients | 27 of 27 identical, worst relative difference **0.000e+00** |
| forward parity | exactly 0.0 |
| state pool after backward's replays | byte-equal in both arms |
| suite | 424 passed, 14 skipped; ruff clean |

## Controls

| control | reading |
|---|---|
| the handed-in state is load-bearing | source-mutating `_gdn` to re-gather instead of using `state_in` moves **26 of 27** gradients, worst rel **8.087e-01** at `layers.1.in_proj_qkv` |
| a pool comparison alone | **does not discriminate** — the re-gathering arm leaves the pool byte-identical, because the scatter converges. Only the gradients see it. |
| the 27-gradient equality is not vacuous | the same assertion is what the mutation arm fails |
| the parity assert can fail at all | flipping `win_parity[0]` after layer 1 fails with `win_parity moved [0, 0] -> [1, 0]`; before the clone fix this control passed silently |
| the batch-2 arm is not decoration | a raise-on-`numel > 1` probe inside `forward` fires with `numel=2`, so the arm reaches a real parity vector rather than the bool-able 1-element case |
| the arm runs on the shipped path | `segment="layer"` selected at `train.py:166` and the full `grpo_loop` tests run: 41 passed (they were 11 red before the parity fix) |
| the selector fires on both sides | `pick(1280) == "mlp"` and `pick(4352) == "layer"` on the expression `_step` uses, plus a source check that the call site reads `_MLP_SEGMENT_MAX_T` rather than a literal. **Two controls, each red by its own assertion:** hardcoding `segment="layer"` fails with `_step must select the segment by T, got segment="layer")`, and moving the threshold to 8192 fails with `T=4352 OOMs with the MLP segment` |
| card state before any peak was read | card 6 free **by UUID**, not by index: 6 compute-app rows, none on `GPU-88e98123-…`; 0 MiB, 0% util, read in the same call as the tree shas |
| the rest of the box, read with every row | all 8 cards' util and memory before, between and after each arm. Cards 1-5 and 7 held **98-100%** from another team's job for the whole session, and a same-code arm elsewhere drifted 133.65 → 140.43 s under it. Card 6 showed `clocks_throttle_reasons.active 0x0` at 1980 MHz, so the coupling is host/PCIe/bandwidth, not thermal. Every row's two arms therefore run back to back in one session |
| the arm was really selected, not assumed | each arm printed its file sha and `segment=` count at launch — `train.py f1b4e2e8ada3` count 0, then `74c258fabda2` count 1; probe `e6d17462707b` count 0, then `ba727b91e3e6` count 1 — and both reverts were `diff`-verified byte-exact after (`TRAIN_ROUNDTRIP_EXACT`, `PROBE_ROUNDTRIP_EXACT`) |
| row 3's answer is its exit status, so the status is captured | `ROW3_EXIT=0`. The first draft piped the probe through `tail`, which reports `tail`'s status and would have made an OOM indistinguishable from quiet output |

## Results

| # | measurement | `segment="mlp"` | `segment="layer"` | verdict |
|---|---|---:|---:|---|
| 1 | gen 1024, `backward_secs`, paired in one session | **69.748 s** | **75.259 s** | **+5.51 s, 1.079x — a real regression** |
| 2 | gen 4096, forward peak | **54.038 GiB** | **14.896 GiB** | **−39.14 GiB, 3.63x smaller** |
| 3 | gen 4096, does the step fit | no (OOM, 290.00 MiB) | **yes** | **the shape that could not run, runs** |

Row 1's `mlp` column: warm mean of steps 2-3, **69.839 / 69.656, spread 0.18 s**; step 0
(70.534) excluded as the JIT step. `layer`: **75.028 / 75.490, spread 0.46 s**; step 0
75.229. Card 6, `c61c1aa`, probe `f1c4b6d6dd86`, `train.py f1b4e2e8ada3` (`segment=` count
0) then `74c258fabda2` (count 1), both printed at launch and the revert `diff`-verified
byte-exact after.

**The regression is 11.9x the largest within-arm spread (5.51 s against 0.46 s), so it is
not noise.** The cost is confined to backward, as the mechanism predicts: `decode_secs`
moves 59.586 → 59.738 (0.25%) and the rollout is untouched, while `step_secs` goes
132.891 → 138.588 (1.043x). Recomputing attention and GDN in every segment replay is work
the MLP-only arrangement did not do.

**Why this row is not 67.77.** The same code on a quiet box measured 67.77 s (#202). This
session's control is **+1.98 s, 1.029x**, with 6 of 8 cards at 98-100% from another team's
job for the whole window. That offset is 36% of the effect being measured, which is why both
columns are measured here rather than one being cited.

Row 2, both arms this session, probe `e6d17462707b` (count 0) then `ba727b91e3e6` (count 1),
64 of 64 segments recorded in each, `ROW2_MLP_EXIT=0` / `ROW2_LAYER_EXIT=0`:

| quantity | `mlp` | `layer` | ratio |
|---|---:|---:|---:|
| forward peak (`max_memory_allocated`) | 54.038 GiB | **14.896 GiB** | 3.63x |
| forward delta (before → end, the quotable form) | 53.968 GiB | 14.826 GiB | 3.64x |
| accumulated over 64 segments | 42.929 GiB | **4.228 GiB** | 10.15x |
| mean rise per segment | 697.8 MiB | **68.7 MiB** | 10.16x |
| live tensors at forward end | 53.984 GiB, 36 shapes | 14.862 GiB, 26 shapes | 3.63x |

**68.7 MiB per segment is below the 85.0 MiB of one retained `[T,hidden]` input** (0.81 of
it), so the layer segment retains less than the MLP one stored — the remaining accumulation
is smaller than `checkpoint`'s own recorded `args`. The 42.929 GiB this entry's predecessor
attributed 88% of to unwrapped activations is now 4.228 GiB, which is the predecessor's
claim confirmed by removal rather than by arithmetic.

### Row 3: the gen-4096 step completes

`ROW3_EXIT=0`, one step, group 8, LoRA-16, micro 1 — **the first time this shape has run on
one card.** #192 concluded cap 4096 was unreachable at group 8; #196 tried to reach it by
removing the T² score matrix and still OOMed at the same 290.00 MiB MLP intermediate. This
reaches it by not keeping 64 layers of attention and GDN activations alive.

```
step_secs   518.528   (step 0: the JIT is inside this number, no warm mean exists)
backward    269.375   51.9%
rollout     249.034   decode 243.463 over 4095 ticks = 59.5 ms/tick
optimizer     0.119
reconciles to 0.0012 s
```

Two things this row is **not**. It is not a warm step — one step means step 0, so the JIT
and the first capture are inside 518.5 s and the number is an upper bound on a warm one. And
it is not a comparison: the `mlp` column is an OOM, so there is no paired time here, only
fits-versus-does-not.

Decode holds at **59.5 ms/tick against 58.4 ms at gen 1024** (1.9%), so the rollout scales
with token count and not with the segment change. Backward goes **75.259 → 269.375 s for 4x
the tokens = 3.58x**, slightly sublinear.

Both columns of every row are measured in this session. The 290.00 MiB OOM in row 3's
`"mlp"` column is the one figure carried over (from the #202 traceback) rather than
re-run — re-OOMing the card to confirm it buys nothing.

**Row 2's `"mlp"` column reproduced the earlier session exactly: 54.038 GiB both times**,
and the accumulation 42.929 GiB both times, on a quiet box then and a box with six cards at
100% now. So a forward peak is repeatable across the contention boundary that moves a time
by 1.029x — which is the byte-count-versus-time distinction holding up under test rather
than being assumed.

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | c61c1aa | H20 card 6 | cuda sm90 | 27B, gen 1024, `mlp` | n/a | n/a | backward 69.748 s/step |
| 2026-09-07 | c61c1aa | H20 card 6 | cuda sm90 | 27B, gen 1024, `layer` | n/a | n/a | backward 75.259 s/step |
| 2026-09-07 | c61c1aa | H20 card 6 | cuda sm90 | 27B, gen 4096, `layer` | n/a | n/a | fwd peak 14.896 GiB, step 518.5 s |

Raw artifacts on the pod: `/work/lcpair.log` (row 1, both arms), `/work/lcpeak.log` and
`/work/lc_peak_mlp.txt` / `/work/lc_peak_layer.txt` (row 2), `/work/lc_row3.txt` (row 3),
`/work/lc_mlp.json` / `/work/lc_layer.json`. Tree `c61c1aa`, stamp and all four file shas
read in the same call as the numbers.

Comparability of row 1: the control is `backward_secs` from `prof_grpo_step.py`,
**not** `train_secs` — `train_secs` wraps the whole `rl_step` including the
optimizer and the sync (`:138-142`), `backward_secs` comes from inside it (`:157`).
The probe's content sha is `f1c4b6d6dd86` on this tree and on `2cafc87`; note that
`36bfe6f`, where the 67.77 s was taken, **does not contain the file** — it was
copied onto a checkout, so the content sha is the comparability-relevant identity
and the tree+path is not reproducible by checkout.

The JIT cache at `/work/tilelang_cache` is warm from another session's arms, so
step 0 here is not a cold number.

**Row 1 needs its own dense control from this session, not the 67.77 s.** All six
readings in the span below were taken while the box was quiet; the box is now
running another team's job on 6 of 8 cards, and a same-code arm drifted from a
133.65 s mean to 140.43 s under it. A `"layer"` reading taken now against a
`"mlp"` reading taken then would measure the neighbours. So row 1 is a **paired**
measurement — both arms back to back in one session on one machine state — and the
67.77 s stays in the table only as the cross-session reference. If the box does not
quiet down, the pair is still valid and the absolute numbers are not comparable to
the earlier entries.

## Decision rule, fixed before the numbers exist

- Row 1 within noise of the paired `"mlp"` reading → `"layer"` becomes the default
  and the `"mlp"` arm is deleted; no two-arm surface survives.

  **The noise band, with its estimator named.** Two steps of one run with identical
  code differ by **0.49 s** (68.26 / 67.77) — that is the within-run spread and it
  is what a paired comparison is judged against. Six dense `backward_secs` readings
  from six trees, all H20 card 6 at this recipe, span **4.61 s**
  (71.53 / 68.30 / 68.26 / 67.77 / 67.31 / 66.92) — that is the cross-tree spread,
  and it bounds only cross-session comparisons. The 71.53→67.77 end of it was itself
  argued to be noise (#190 at 0.947x), so quoting the full span as an error bar
  partly cites a conclusion as its own evidence. Six points from six trees are a
  range, not a sample, so no confidence interval is computed from them.

  Since row 1 is paired in one session, the band that applies is the **0.49 s**
  within-run figure, not the 4.61 s span.
- Row 1 regresses and row 3 fits → default stays `"mlp"` and the caller picks by
  shape, with the threshold measured on live activation bytes at T **on both
  arms**, not derived from one. **← this branch fired.**
- Row 3 still does not fit → the whole thing comes out, and this entry's finding is
  that one card is the wrong shape for cap 4096.

**What the threshold actually is, versus what this rule asked for.** The rule wanted a
threshold on live activation bytes at T, measured on both arms. What shipped is a T
threshold at the bracket's low end: `_MLP_SEGMENT_MAX_T = 1280`, from two shapes rather
than a curve — 1280 runs both ways and MLP is 1.079x cheaper, 4352 runs only as `"layer"`.
Deliberate, and cheaper than what the rule asked for: a bytes model needs a sweep to
calibrate and would still be a model of the quantity rather than the quantity. The cost of
the shortcut is named in the code — shapes in 1280..4352 pay 1.079x that a measured
crossover might avoid — and the ponytail line says which measurement moves it (both arms at
2048 and 3072).

## Rule

**A checkpoint segment's boundary is a shape decision, not a correctness one, and the two
directions do not trade off against each other.** Widening the segment from the MLP to the
whole layer cut the forward peak 3.63x and made a shape run that had OOMed twice under two
different diagnoses — and cost 1.079x on backward at a shape that already worked. Neither
number argues against the other; they argue for a selector. The mistake available here was
to read the 3.63x as a win and flip the default, which would have taxed every training shape
we actually run to buy a shape we do not run yet.

Corollary on how the bracket was reached: **the two arms had to be measured in one session.**
The same control read 67.77 s on a quiet box and 69.748 s with six neighbouring cards at
100% — a 1.029x offset, 36% of the effect. Citing the earlier figure would have reported
1.11x instead of 1.079x, and the direction would still have been right, which is what makes
that class of error survive.
