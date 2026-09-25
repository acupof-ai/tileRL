# Two NVFP4 loader limits: cross-shard scales and a 40.7 GB bf16-lm_head pack — 2026-09-25

> Status: fixed by `bb3fe08b` (PR pending). The V100 had the checkpoint patched
> out of tree to unblock the ThinkingCap cutover; both workarounds are
> superseded by the loader fix. The earlier CPU claim that "main needs zero
> loader changes to load this checkpoint" is withdrawn — it did not cover
> either defect.

## What happened

Cutting V100 production to `bottlecapai/ThinkingCap-Qwen3.8-27B-NVFP4`
(ModelScope; the HF mirror returns 403) failed twice on `load_hf`, both in
`src/tilerl/model.py`, tree `5a0c54cc`.

### 1. A packed weight's scales can live in a different shard

The loader iterated shard by shard and read each `.weight_packed`'s
`weight_scale` / `weight_global_scale` from the same file's tensor dict. This
checkpoint splits exactly one triple across files:
`model.language_model.layers.22.mlp.up_proj.weight_global_scale` (a scalar
float32) was in `model-00001-of-00002.safetensors` while its `.weight_packed`
and `.weight_scale` were in shard 2. A scan of the index: 399 same-shard
triples, **1 cross-shard, 0 missing**. The load died with a KeyError naming
the global-scale tensor. The loader never looked at the index for siblings —
the index maps every tensor to its file and had the answer.

Fix: a `sibling()` helper in `load_hf` follows `model.safetensors.index.json`
to the owning shard when a scale is not in the packed tensor's file. A sibling
that is neither in the file nor routable via the index still KeyErrors (the
negative control in the gate).

### 2. An unpacked bf16 lm_head packed whole needs 40.7 GB of host RAM

`lm_head` is on the NVFP4 quant ignore list (`recipe.yaml`:
`ignore: [lm_head, 're:.*mtp.*', ...]`), so the checkpoint ships it as bf16
(248320×5120). The loader's "pack the bf16 linears the checkpoint did not
ship quantized" pass called `pack_fp4` on it in one shot. The reference
`pack_fp4` builds an e2m1 distance tensor `[n, K/B, B, 8]` of f32 —
`248320·5120·32 bytes = 40.7 GB` (37.9 GiB) for that one tensor — and the V100
has 31 GiB host RAM. (The chunk sizer's 48 B/weight is a separate combined
upper bound: 32 B for that distance tensor plus at most 16 B for the other
live temporaries — master float copy, scaled input, index bytes; it is not the
distance tensor's own size.) The shipped Qwen3.8 checkpoint never hit this:
its lm_head is already `.wq/.scale/.oscale`.

Fix: `_pack_fp4_bounded` packs row chunks under a byte budget
(`TILERL_FP4_PACK_BUDGET_BYTES`, 512 MiB default). Blocking is along K, so row
chunks are independent: pack and renorm are per-row operations, and the
concatenated result is **bit-identical** to one whole pack (proved, not
asserted — see gates).

## Gates

CPU (`tests/test_weights.py`, all three fail on the unpatched loader):

- `test_nvfp4_sibling_scale_in_another_shard_loads` — a two-shard fixture with
  one triple split; loaded served bytes equal `_native_fp4(packed, scale,
  gscale, divide=True)`; removing the index routing raises `KeyError`.
- `test_pack_fp4_bounded_is_bit_identical_and_respects_budget` — chunked and
  one-shot packs are `torch.equal`; the held budget forces multiple calls.
- `test_load_hf_packs_bf16_linears_under_a_tiny_budget` — end-to-end tiny load
  at a 6 KiB budget; every observed `pack_fp4` call's distance-tensor footprint
  stayed within it and the served bytes equal one whole pack. A loader that
  ignored the budget makes exactly one call per linear and trips the call-count
  half.

V100 sm70, pristine checkpoint in `~/models/ThinkingCap-orig` (re-downloaded
from ModelScope, sizes match `~/dl_tc_ranges.log`, no edits):

- truncated load (`num_layers=1`, host-only) packs lm_head bounded and the
  result is **bit-identical to the independently written offline chunk-pack**
  now living in the modified production checkpoint (`lm_head.wq/scale/oscale`,
  `(248320, 2560)`): `LMHEAD_BITIDENTICAL`.
- full load of the pristine dir succeeds with the production ThinkingCap
  service still running: the one cross-shard triple loads
  (`wq (17408, 2560)`, `scale (17408, 320)`), **497/497** fp4 linears packed,
  **0** bf16 masters retained.

## Workarounds superseded

Two card-local scripts unblocked the same-night cutover by editing the model
directory: `~/fix_tc_shard.py` (moves the cross-shard scalar into shard 2) and
`~/pack_tc_lmhead.py` (offline chunk pre-pack into served layout). Both are
repository-external hidden state — a fresh download breaks again — and are the
reason this code fix exists. The production directory
`~/models/ThinkingCap-Qwen3.8-27B-NVFP4` still holds those patched bytes; the
pristine copy is `~/models/ThinkingCap-orig` and the loader now loads it as
shipped.
