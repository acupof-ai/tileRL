# A layer-wide checkpoint segment — H20 sm90, 2026-09-07

> Status: **measurement pending** — the CPU half is green and the three card rows
> below are empty on purpose. No default has moved.

## Context

`autograd.checkpoint` wrapped `_mlp_body` alone, so every layer's attention and
GDN activations stayed on the tape for the whole forward:
[the MLP-only measurement](2026-09-06-checkpointing-covers-the-mlp-only.md) put
**697.8 MiB retained per segment against the 85.0 MiB the wrapper stores**, i.e.
the checkpoint explained 12% of a 42.929 GiB accumulation. This entry is the
obvious follow-up — make the segment the whole layer — and its point is whether
the obvious thing is worth defaulting to.

`forward(..., segment=)` takes `"mlp"` (unchanged default) or `"layer"`.

**How an arm is selected, since nothing outside the tests passes the flag.**
`git grep 'segment='` finds only `tests/test_e2e.py`; the two forward calls that
reach a tape are `train.py:155` (the shipped training path, via
`_step` → `run`) and `scripts/prof_forward_memory.py:138`. So the card arms are
selected by editing those two lines in the measured tree — rows 1 and 3 at
`train.py:155`, row 2 in the probe — not by a flag threaded through
`rl_step`/`grpo_loop`/`cli`. That is deliberate: `train.py:155` is the one line
that changes permanently if `"layer"` wins, and a plumbing chain added for the
measurement would be deleted by the same PR that wins. The measured arm is
therefore the shipped training path with one word changed, not a probe-local
monkeypatch.

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
`train.py:155` and running the real `grpo_loop`: **11 tests in `test_rl.py` went
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
| the arm runs on the shipped path | `segment="layer"` selected at `train.py:155` and the full `grpo_loop` tests run: 41 passed (they were 11 red before the parity fix) |
| card state before any peak is read | (pending) `nvidia-smi --query-compute-apps=pid,used_memory` returning zero rows, read in the same call as the numbers |
| the rest of the box, read with the card | (pending) all 8 cards' util and memory. Another team's job took cards 1-5 and 7 to 100% during the arms above, and a same-code run drifted 133.65 → 140.43 s under it. Card 6 itself showed 59% util, 1980 MHz, `clocks_throttle_reasons.active 0x0`, so the coupling is host/PCIe/bandwidth, not thermal. **Row 1 is a time and is not comparable across that boundary**; rows 2-3 are byte counts, which contention does not change, but the allocator's behaviour under a busy host is not something to assume. |

## Results

| # | measurement | `segment="mlp"` | `segment="layer"` | verdict |
|---|---|---:|---:|---|
| 1 | gen 1024, `backward_secs`, paired in one session | | | |
| 2 | gen 4096, forward peak | | | |
| 3 | gen 4096, does the step fit | | | |

Both columns of every row are measured in this session. The earlier figures —
67.77 s backward, 54.038 GiB forward peak, and the 290.00 MiB OOM — are the
cross-session reference, not the `"mlp"` column: they were taken on a quiet box and
row 1 is a time.

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-07 | 7538517 | H20 card 6 | cuda sm90 | 27B | n/a | n/a | pending |

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
  arms**, not derived from one.
- Row 3 still does not fit → the whole thing comes out, and this entry's finding is
  that one card is the wrong shape for cap 4096.

## Rule

(pending the numbers — a rule written before the measurement would be the
prediction, not the finding)
