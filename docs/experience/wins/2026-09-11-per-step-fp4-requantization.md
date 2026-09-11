# Per-step re-quantization: a full-SFT step refreshes the served fp4 faces — 2026-09-11

> Status: **shipped** — tiny CPU gates (8) and the on-card sm90 confirmation on
> the real 27B (H20 card 0, commit `c66c472f`): twiddle-layout correct, served
> parity 0.0021, repack 6.27 s/step. Device-resident shared-engine decode
> (serve and train in one process) is still a later integration.

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
- **Arch layout is re-applied, not assumed natural.** A card's `materialize`
  rewrites a `.wq` into its decode layout once and tags it (`_tl_layout` =
  `tw-bf16` on sm90, `tw-f16` on sm70); sm70 also narrows the scale plane to
  f16. A fresh `pack_fp4` is NATURAL nibbles, so the re-pack runs the SAME
  reference rewrite the slot's tag names (`reference.FP4_LAYOUT_TWIDDLE`, the
  single map `materialize` also reads) and copies the scale at the slot's dtype
  before `copy_`. Without it, after the first step sm90/sm70 decode feeds
  natural bytes to a twiddled-layout kernel — silent, and RefBackend (identity
  materialize) makes every CPU gate blind to it.
- `copy_`, never rebind: every `.wq/.scale/.oscale` keeps its address, so a
  captured decode graph — which baked those addresses — keeps serving the new
  weights. This is the same property `invalidate_weights` relies on for the
  optimizer's in-place `p.copy_`; no graph recapture is forced.
- Re-packing zero slots is a wiring error (faces dropped, or non-fp4 config)
  and raises rather than returning 0.

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

Parity is asserted by **exact argmax plus a distribution bound**, not a single
`allclose` band — fp4 quantization error is real and a hand-picked band is the
wrong instrument. Effective relative error is `|served−master|/|master|` per
logit with no floor. Measured on the fixture:

| arm | p99 eff-rel | max abs | argmax |
|---|---:|---:|---:|
| correct repack | **0.150** | **0.152** | **64/64** |
| wrong-block mutant (block 16 into the block-32 face) | 0.444 | 0.314 | 64/64 |

The bounds are read from the GAP between those two rows (p99 < 0.25, max abs <
0.22): correct passes, the block-16/32 mixup fails both — and note its argmax
stays 64/64, which is exactly why token equality alone cannot police the bytes.

- `test_repacked_served_logits_match_the_bf16_master` — exact argmax + p99/max
  distribution bound.
- `test_the_distribution_bound_discriminates_a_wrong_block_repack` — the
  block-16-into-32 mutant keeps argmax exact but breaks both bounds (premise
  asserted first), so the bound is proved to catch what argmax cannot.
- `test_a_step_refreshes_the_served_slot_bytes_in_place` — after steps the
  `.wq/.scale/.oscale` bytes change, at the SAME `data_ptr`, and `.wq` is
  byte-equal to a fresh `pack_fp4` of the trained master.
- `test_a_twiddled_slot_is_repacked_through_the_same_twiddle` — a slot tagged
  `tw-bf16` (as sm90 materialize leaves it) equals `twiddle_fp4(pack_fp4(m))`
  after a step and NOT the natural pack; red on the pre-fix code.
- `test_requant_with_no_served_slots_raises` — n==0 raises.
- `test_stale_fp4_path_leaves_the_served_bytes_untouched` — negative control:
  skip the hook and the slots stay byte-identical to step 0 while the masters
  trained.
- `test_two_steps_served_tokens_equal_the_trained_master_fixed_seed` — fixed
  seed, two steps: a decode off the served face picks exactly the trained
  bf16 master's token sequence.
- `test_stale_and_repacked_streams_diverge_token_level` — a strong arm
  (10 steps, lr 0.2) where the stale and repacked served streams pick
  different tokens; the off-policy failure at token level.

Mutants: (a) delete the streaming-branch `post_step()` call → slot-byte and
token-divergence gates red; (b) delete the tag-named twiddle rewrite → the
twiddled-slot gate red (the sm90 defect 52 reproduced on CPU).

## Rule

A weight face served beside a trained master is a cache: after an in-place
update it must be re-derived into the same buffers — INCLUDING the arch byte
layout the serving kernel reads, not just fresh natural nibbles — with
addresses preserved, or the serving engine silently decodes the pre-step
policy. Gate it at the bytes (slot == the correctly-laid-out fresh repack, same
pointer), at the error distribution (bound in the measured gap to a wrong
block), and at fixed-seed tokens, with the stale path as the negative
control — argmax alone neither moves on short steps nor catches a block mixup.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | local (fp4 tiny) | cpu | 8 gates green; two mutants red on their intended gates |
| 2026-09-11 | H20 card 0 (`c66c472f`, real 27B) | cuda sm90 | see below: twiddle layout 0 mismatch, parity 0.0021, repack 6.27 s/step |

## On the real 27B (H20 card 0, sm90)

`scripts/probe_requant_card.py` (commit `c66c472f…`, `TILERL_TARGET=cuda
TILERL_27B_CKPT=/work/Qwen3.8-27B-NVFP4`, card 0, 172 s `load_hf`), card name
NVIDIA H20. Verbatim output:

```
arch sm90 device NVIDIA H20
fp4 served keys: 264
tagged slots: 264 of 264
sample tag layers.0.down_proj tw-bf16
requant keys 264 per-step repack 6265.718 ms
layout mismatches (of 16 sampled): 0
served-vs-ref max rel per linear: ['0.0015','0.0019','0.0019','0.0013','0.0015',
  '0.0013','0.0019','0.0018','0.0021','0.0017','0.0013','0.0017']
worst max rel 0.0021, argmax rows agree 12/12
PROBE_OK
```

- The sm90 `materialize` twiddle+tag premise is real: **all 264** fp4 slots are
  `tw-bf16`. After a perturb + `requantize_fp4`, every sampled slot equals
  `twiddle_fp4(pack_fp4(master))` with the tag preserved — the exact defect
  (natural nibbles fed to a twiddled decode kernel) does not occur.
- Through the actual sm90 `linear_fp4` kernel, served logits vs the natural f32
  reference agree to **worst max-rel 0.0021** over 12 sampled linears, with
  **12/12 argmax rows** — an order of magnitude tighter than the tiny fixture.
- Re-packing all 264 keys costs **6.27 s/step**. The ponytail is the touched-key
  set: a full SFT step updates every key, but an LoRA-free incremental or a
  layer-chunked training path would repack only what moved.


