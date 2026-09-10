# Cost model

One primitive prices every byte and kernel; card, host and SSD occupancy
derives from it, consumed by the allocator. The invariant is

```
peak = sum(static rows, derived) + transient
```

`derived == measured` on every static row; `transient` (partials, workspace,
graph pools, slack) is the measured remainder, its own row, never a tolerance.

## Format

`precision.Format` is bits per element plus scale planes `(group, dtype)`:
`group` is an int for one scale per that many elements on the last axis, a
tuple over trailing axes (`None` = that whole axis), or bare `None` for one per
tensor. Count = leading dims × Π ceil(trailing/group): per-16 `(16,)`, per-row
`((None,),)`, an `[N/128,K/128]` grid `(128,128)`.

```
bf16      = Format(bits=16)
f32       = Format(bits=32)
fp8_kv    = Format(bits=8,  scales=((head_dim, f32),))         # one f32 per plane x head x token
nvfp4     = Format(bits=4,  scales=((16, e4m3), (None, f32)))  # disk: ModelOpt packing
nvfp4_dev = Format(bits=4,  scales=((16, f32), ((None,), f32)))# device: f32 scales, f32/row
nvfp4_dev_b32 = Format(bits=4, scales=((32, f32), ((None,), f32)))  # bf16 repacked at load (pack_fp4 block 32)
fp8_block_dev = Format(bits=8, scales=(((128,128), f32),))               # fp8 block grid, no row scale
fp8_dev   = Format(bits=8,  scales=(((128,128), f32), ((None,), f32)))  # fp8 grid + f32/row
nbytes(fmt, shape) = numel * bits // 8 + sum(scale_count * itemsize for each plane)
```

`nbytes` is the only byte arithmetic in the tree: `kv_cache.bytes_per_token`,
the `num_blocks` fit, the draft pool charge and the ISO frame budget call it;
none carries its own `element_size()` product. Device faces differ from disk:
`renorm_fp4_scale` widens the block scale to f32 and splits the global into a
per-row epilogue; the fp8 block grid is `[N/128,K/128]`. The **weights row comes
from the checkpoint index, not config** — the 27B checkpoint is 168 on-disk
block-16 nvfp4 + 96 bf16 repacked at load block 32 = 264 served nvfp4 faces,
plus 233 fp8, and a key's face depends on its tensor name.
`precision.checkpoint_weight_specs(dir)` classifies the safetensors headers
(shapes only); the SERVED map `model.checkpoint_weight_faces(cfg, dir)` applies
load_hf's key rules (vision/MTP dropped) and maps a bf16 `fp4_param_keys` linear
to `nvfp4_dev_b32` under cfg.fp4.

Checks: tiny-model derived pool bytes equal `k_pool/v_pool/k_scale/v_scale`
storage to the byte under bf16 and fp8; device-face Formats equal served tensor
storage (`pack_fp4`+renorm for nvfp4, fp8 GEMM operands). At 16 planes x 4
heads x 256: 64 KiB/token bf16, 32 KiB + 512 B fp8. The 27B resident total is
**24,436,981,888 B (24.44 GB)** — the exact header-only
`memory.weight_row_faces(checkpoint_weight_faces(cfg, dir))` sum (raw
`checkpoint_weight_specs` over-counts non-served tensors); no GPU needed. An
all-fp4 config derivation gives ~15 GB and is known-wrong: it misses the fp8
population.

## Plan

`memory.plan(cfg, params, device_free, *, num_slots, num_blocks, ...) -> list[Row]`
is a pure function laying the device rows out before anything is allocated:

```
Row(tier, owner, bytes, note="")        # bytes already computed through nbytes
tier  in {device, host, ssd}
owner: weights, state_slots, kv_pool, draft_pool (held), plus *_budget rules
```

The Row carries the byte total, not fmt/shape/count — those feed the `nbytes`
call. Budget rules (`free*2/3` pool, `free/4` for `state_bytes`, `dram_bytes`,
`BLOCK_TOKENS`) emit their own `kv_pool_budget`/`prefix_entries_budget` rows so
a sum over allocations skips them. `build_engine` allocates from the rows:
`fit_num_blocks` fits `num_blocks`; the draft pool is its OWN row
(`per_kv_block_bytes` excludes it, no double count). The pool is
`num_blocks x per_block`, the fp8 scale plane inside `per_block`; a request's
occupancy is `blocks_used x per_block`. A device prefix entry is pool blocks
plus a state slot already counted — a row only when demoted to `host`/`ssd`.

The training engine is a separate `memory.train_plan` over the same `nbytes`:
`adapter` (bf16), AdamW `optimizer_state` (2 f32 per trained param), ISO `frame`
(host-tier f32 U,S,V per 2-D weight, with Adafactor factors on the frames), and
the `tape` row — what RecordingBackend keeps after one layer-segmented forward
(embedding, one boundary hidden per layer, final norm, head output; +adapter
head entries); `tilerl train --dry-run --recipe X` prints those rows through
the same `memory_table` renderer without a build.

`memory.memory_table` adds a measured column and closes the peak: the measured
column is each owner's materialized tensor-storage sum on every target; on cuda
the peak is `torch.cuda.max_memory_allocated` and a final `transient` row is
`peak − Σ static`, so `peak = sum(static) + transient` to the integer;
transient is suppressed without the peak measurement.
`tilerl serve --dry-run` prints the table cardless; `--checkpoint DIR` prices
the weights row header-only from `checkpoint_weight_faces` (off cuda pass
`--device-free`); `--record-residency` (cuda) adds measured residency. Gate:
`derived == measured` for `weights` and `kv_pool` on the tiny model; the
training rows equal the live adapter / optimizer / frame tensor storage and a
counted tiny tape (`tests/test_train_plan.py`); a nonzero static-row delta on
27B is an error entry, not a tolerance. Nothing runs in the tick — `plan` is
build-time arithmetic, the table reads counters.

## Kernel cost

Each launched kernel declares a pure helper next to its registry entry:

```
(cfg, tick) -> (bytes, flops)   # per kernel; bytes via nbytes(fmt, ...), row keys "bytes"/"flops"
bound = max(bytes / bandwidth, flops / peak)
```

`bandwidth` and `peak` are one measured calibration row per card —
`hbm_bw_gbs` (D2D copy, read+write) and `bf16_peak_tflops` (one big GEMM),
written by `tilerl bench --calibrate --card N` through the benchrec validator;
never a datasheet number. A kernels mode of `tilerl bench` prints
`kernel / shape / face / bytes / flops / bound / ms / % of bound` for one 27B
tick's kernels; the tick's cost is their sum. `--checkpoint DIR` prices every
linear at the face `checkpoint_weight_faces` derives (including load_hf's
bf16→fp4 repacking at pack_fp4 block 32); without it all linears take the
config nvfp4 face. The mixed checkpoint makes the checkpoint-priced tick
22.36/25.63 GB at B=1/B=8 against 14.88/18.16 GB all-nvfp4. Each GEMM's ms is
timed through the kernel its OWN face resolves to (linear_fp4/linear_fp8),
never a bf16 surrogate; `--prefill S` adds the prefill rows (one GEMM over
M=S tokens, K/V written and read, chunked GDN forward; S=4096 = 155.26 GB /
204.5 TFLOP). Without a card or calibration row, measured columns read
`pending-remote` and only the bound is shown. Gates: decode K/V read and
prefill K/V write each equal pool bytes through the same `nbytes`; GDN chunk
matmul dims are cfg-derived; a mixed tiny checkpoint reaches the table per
linear at its own device face (`tests/test_kernel_cost.py`,
`tests/test_calibration.py`).

## What this replaced

The three hand-rolled byte products (`kv_cache.py`, `engine.py` twice) moved to
`nbytes`/`memory` (#458); 17 superseded probes deleted, three re-derived on
device-face denominators (#461, #470); roofline/prefill/checkpoint faces/
calibration in #457/#463/#466/#468; plan, checkpoint dry-run and the
transient-peak close in #460/#465/#469. Percent-of-HBM claims the model cannot
reproduce are corrected or stale; surviving card-only measurements stay in
their wins/errors entries with a one-line rerun command.

## Ownership

| Unit | Owner | PRs |
|------|-------|-----|
| `Format`/`nbytes`, call sites, checkpoint faces, byte gate | cc | #458, #462 |
| `plan`, dry-run, residency, transient peak | 52 | #460, #465, #469 |
| kernel roofline, prefill rows, calibration | 5f | #457, #463, #466, #468 |
| recompute recorded numbers, delete superseded probes | 65 | #461, #470 |
