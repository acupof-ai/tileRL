# Per-step re-quantization: a full-SFT step refreshes the served fp4 faces — 2026-09-11

> Status: **tiny CPU gates green** (fp4 fixture); 27B served-bytes accounting and
> GPU timing stay pending-remote per the card recall.

## Context

Full-parameter SFT (`_train_full`) loads with `keep_master=True` and immediately
`drop_quantized(model)`: it trains bf16 masters and frees every served
`.wq/.scale/.oscale` face, because the trainer never reads them. The P3 OPD
shape is one runtime serving AND training — after a step the serving engine
must decode the new weights without a reload or a graph recapture. Nothing kept
the served bytes in step with the masters.

## What worked

`model.requantize_fp4(model)` re-packs every trained bf16 fp4 master into its
**existing** served slots after an optimizer step:

- `pack_fp4(master, block=…) + renorm_fp4_scale(scale)` — the exact load-time
  pair in `load_hf`, so a served face is byte-identical to a fresh load of the
  trained master.
- The block size is read off the slot (`K / scale.shape[1]`): a bf16 linear
  repacked at load is block 32 (`nvfp4_dev_b32`), an on-disk NVFP4 linear is
  block 16. A fixed block 32 would not copy into a block-16 slot.
- `copy_`, never rebind: every `.wq/.scale/.oscale` keeps its address, so a
  captured decode graph — which baked those addresses — keeps serving the new
  weights. This is the same property `invalidate_weights` relies on for the
  optimizer's in-place `p.copy_`; no graph recapture is forced.

The hook is one optional `post_step` callable on `train_step` / `_step`,
invoked once after a finite update in both the streaming (`step_one`) and
non-streaming (`optimizer.step`) branches. `tilerl train --served-fp4`
(full-parameter SFT only, default off) keeps the served faces and sets the
hook to `requantize_fp4(model)`. The flag refuses a non-fp4 config and a LoRA
run (LoRA keeps the frozen faces). Off flag, the path is unchanged: masters
only, faces dropped.

In scope are the fp4 faces only (`pack_fp4`/`renorm`). The checkpoint's
fp8-native linears train from a bf16 master but their `.w8/.wscale` faces are
not refreshed by this hook.
# ponytail: fp8 face refresh is a different packer (`fp8_block_dev`); add it
# when a measured shared-engine run serves an fp8 layer after a step.

## Gates (`tests/test_requant_fp4.py`, fp4 tiny at the 27B's ~0.02 weight scale)

- `test_repacked_served_logits_match_the_bf16_master` — served forward
  `allclose(rtol=5e-2, atol=0.1)` to the bf16-master forward and identical
  argmax. Random N(0,1) tiny weights saturate the e2m1 grid (fp4 meaningless
  there); the fixture scales linears to the real ~0.02 magnitude.
- `test_a_step_refreshes_the_served_slot_bytes_in_place` — after steps the
  `.wq/.scale/.oscale` bytes change, at the SAME `data_ptr`, and `.wq` is
  byte-equal to a fresh `pack_fp4` of the trained master.
- `test_stale_fp4_path_leaves_the_served_bytes_untouched` — negative control:
  skip the hook and the slots stay byte-identical to step 0 while the masters
  trained.
- `test_two_steps_served_tokens_equal_the_trained_master_fixed_seed` — fixed
  seed, two steps: a decode off the served face picks exactly the trained
  bf16 master's token sequence.
- `test_stale_and_repacked_streams_diverge_token_level` — a strong arm
  (10 steps, lr 0.2) where the stale and repacked served streams pick
  different tokens; the off-policy failure at token level.

Mutant (delete the streaming-branch `post_step()` call): the slot-byte and
token-divergence gates go red; parity gates stay green as they should.

## Rule

A weight face served beside a trained master is a cache: after an in-place
update it must be re-derived into the same buffers (addresses preserved), or
the serving engine silently decodes the pre-step policy. Gate it at the bytes
(slot == fresh repack, same pointer) and at fixed-seed tokens, with the stale
path as the negative control — argmax alone does not move on short steps.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | local (fp4 tiny) | cpu | 5 gates green; mutant red on 2 |
| 2026-09-11 | 27B H20 | pending-remote | per-step repack cost, parity on real weights, shared-engine decode |
