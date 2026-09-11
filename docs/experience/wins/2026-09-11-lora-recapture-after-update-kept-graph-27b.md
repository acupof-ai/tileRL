# P2.0 step 0 on a card: the kept decode graph replays an in-place LoRA update, tokens bit-equal to eager — H20 sm90, 27B NVFP4, 2026-09-11

> Status: card-measured (`scripts/recapture_lora.py`, H20 card 4). Full-parameter
> SFT does not coexist with the captured engine on one card (separate errors
> entry); the LoRA path — what P1 trains — passes.

## Context

P2.0 asks whether a serving engine can **keep its captured decode graph across
an optimizer step** and replay the updated weights without rebuilding. On-policy
RL refuses a model carrying a stale graph or stale prefix KV, so today every
training step tears the graph down and pays a re-capture. The recapture premise
is narrow: the optimizer updates every tensor **in place** (`AdamW.step_one`
ends in `p.copy_()`), so the addresses a captured graph baked survive; only the
f32 cast cache must be refilled (`invalidate_weights`).

The first probe ran full-parameter SFT and OOMed at the backward on a free
95.22 GiB H20 (weights + bf16 masters + moments + tape) — that is a footprint
property, not a bug; see
`errors/2026-09-11-full-sft-oom-does-not-coexist-with-serving-engine.md`.
LoRA is the path P1 actually uses: a frozen fp4 base plus a rank-16 adapter
(~124.8M params), updated by the same in-place AdamW.

## The gate

`scripts/recapture_lora.py` builds two engines sequentially from the 27B NVFP4
checkpoint (`/work/tilerl-ckpt/Qwen3.8-27B-NVFP4`), attaches a seed-0 rank-16
adapter **after** `build_engine` in each, and takes one fixed-seed LoRA SFT step
(seq 256, lr 0.02):

1. **captured** — graphs on; warm/capture; record `base`; assert a graph was
   held; `invalidate_weights` with NO step must leave the rollout unchanged;
   one LoRA step; `invalidate_weights`; record `after`; assert the graph is
   still held (kept, not rebuilt).
2. **eager** — graphs off; the same-seed adapter and the same data step; record
   `after_eager`.

The core claim is `after == after_eager`. Verbatim from the card:

```json
{
  "graphs_held_before": 1,
  "graphs_held_after": 1,
  "no_step_unchanged": true,
  "update_changed_tokens": true,
  "recaptured_equals_eager": true,
  "captured_secs": 29.0,
  "captured_decode_ms_median": 24.4,
  "captured_decode_ms_min": 24.29,
  "eager_secs": 41.3,
  "eager_decode_ms_median": 158.18,
  "eager_decode_ms_min": 157.1,
  "verdict": "PASS"
}
```

## Two distinct clocks (do not conflate)

The end-to-end phase times (`captured_secs 29.0`, `eager_secs 41.3`) include
load, JIT, graph capture, the training step, and several rollouts — they are
**not** the P2.0 number and are not in favor of the graph (the captured phase
builds the replay pool). The wall claim is the **steady-state post-step decode
tick** on identical post-step LoRA weights, timed over a decode-only window
with the prefill and first ticks excluded:

- captured graph: **24.4 ms/tick median** (24.29 min)
- eager: **158.2 ms/tick median** (157.1 min)
- **6.5× per decode tick**, post-step, same weights.

Keeping the graph across the update costs nothing at replay and preserves the
eager-vs-graph gap after a real optimizer step — the reason on-policy RL can
keep decoding fast without a re-capture. (The same B=1 shape appeared in the
decode-roofline probe, ~12 ms graph vs ~49 ms eager; batch-8 graph replay ties
eager, a separate padding/width defect tracked in the decode-gap write-up.)

## Two traps the run surfaced (both now hardened)

1. **`add_lora` must run after `build_engine`.** Building calls
   `backend.materialize`, which replaces every param moving device/dtype with a
   new tensor (new `id()`). An adapter attached before points at the
   pre-materialize CPU tensors; the forward reads the new device tensors, so
   every adapter is silently absent and `train_step` asserts *"tape produced no
   parameter gradients."* `build_engine` now raises if materialize orphaned a
   pre-build adapter, with a hermetic red test
   (`tests/test_lora_build_order.py`; a `RefBackend` whose materialize clones
   stands in for the GPU move — `RefBackend` itself is the identity, so raw-model
   CPU tests stay green).
2. **The step has to move the argmax while equality is what is gated.** rank-16 /
   seq-64 / lr-1e-4 trained but did not change greedy tokens; seq-256 / lr-0.02
   did. `update_changed_tokens` is a real control — a gate that only compared
   captured to eager would pass identically even if the step did nothing.

## Rule

A decode graph can be kept across an in-place optimizer step when every updated
tensor is updated by `copy_()` into the address the graph holds — proven for the
LoRA path on the 27B, post-step captured tokens bit-equal to eager, 6.5× the
eager steady tick. Attach adapters after materialize, refill the f32 casts
after the step, and gate both equality (captured==eager) and that the update
actually changed the rollout. Full-parameter coexistence is a separate process,
not a one-card engine.
