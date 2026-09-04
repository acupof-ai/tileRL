---
question: Does tileRL run on Apple Silicon, and how fast is the 27B there?
status: measured
source: tileRL, M4 Pro (48 GB unified), 2026-09-04; TILERL_TARGET=metal, torch 2.13.0, tilelang 0.1.13
---

# Metal runs the suite green, and its gemms are 28x off torch-eager

The Metal target was one bug from a green suite and is now green: **256 passed,
9 skipped, 0 failed**. The 27B does not run here, and the reason is not memory.

## The bug: `_c` checked contiguity but not offset

`test_block_draft_matches_reference` failed with

```
RuntimeError: kernel gemm_nt input A byte_offset violates packed ABI constraint;
              expected: 0, got: 256
```

`Backend._c` (backend.py:259) is the one helper every kernel argument passes
through, and it read:

```python
return t if t.is_contiguous() else t.contiguous()
```

A **row slice is contiguous and starts partway into its storage**. `x[1:2]` of a
`[4, 64]` f32 tensor is contiguous with `storage_offset=64` — 256 bytes, exactly
what the error reports. CUDA tolerates a non-zero `byte_offset`; the Metal ABI
rejects it.

The fix that suggests itself does not work. **`.contiguous()` is a no-op on such
a view**, because the view already is contiguous — measured:

```
x[1:2].contiguous()   offset=64   <- unchanged
x[1:2].clone()        offset=0
x[:, 1:3].contiguous() offset=0   <- resets, because this view is NOT contiguous
```

So `_c` needs a copy, not a `.contiguous()` — and the copy is charged only to the
target that needs it, because the next section measures that CUDA would pay for
it too:

```python
if t.is_contiguous():
    return t.clone() if self._zero_offset and t.storage_offset() else t
return t.contiguous()
```

`_zero_offset` is `arch == "metal"`, set beside `io` and `scale_io` in
`__init__`. One helper, 49 call sites, no per-call-site `.contiguous()`.

**Negative control:** reverting `_c` (with `__pycache__` cleared) brings back
exactly the three failures and nothing else — 3 failed / 9 passed in
`test_dflash2.py`, against 12 passed with the fix. Forcing `_zero_offset = False`
on metal reproduces the same three, so the gate is what carries the fix.

## The clone is not free on sm90, which is why it is gated

The first version of this fix cloned unconditionally. CUDA *tolerates* the
non-zero offset, so that would have been a new device-to-device copy on a helper
49 call sites reach, on a target that never needed it. Counted with
`scripts/probe_c_offset.py`, one tiny-model decode tick:

| target | contiguous, offset==0 | **contiguous, offset!=0** | non-contiguous |
|---|---:|---:|---:|
| cpu | 224 | **0** | 8 |
| metal | 224 | **0** | 8 |
| **sm90 (H20)** | 158 | **6** | 6 |

sm90 is not zero. Six `[1, 1, 32]` f32 views at offsets 32 and 64, and a
`[1,1,32]` clone measures **6.68 us** on the H20 — **40 us per tick**, 0.35% of an
11.4 ms B=1 decode tick. Under the 0.97x gate, but not nothing, and not a cost
CUDA owes. With the gate, sm90 reports `zero_offset=False` and clones none of
them.

The offset views come from `h[:, 1:]` on the draft path (dflash2.py:153, :229),
which drops the first position: `test_dflash2.py` alone produces **76** of them,
all `[3, 64]` f32, via `model.py:177 _base_linear -> backend.py:427 linear ->
_rows`.

## The second failure `-x` hid

Running without `-x` found a fourth failure the truncated run never reached:
`test_fused_projections_parity` asserted `torch.equal(y2, y3)` — bit-identity
between one fused gemm and two separate ones. Their reduction orders coincide on
cpu and do not on Metal: **9.5e-06 apart**. The assertion was wrong, not the
fusion. Now `allclose(rtol=0, atol=1e-4)`; a broken fusion is off by orders of
magnitude, not 1e-05. Passes on both cpu and metal.

## Speed: 0.097 tok/s of linears alone, 28x off torch-eager

Per-token cost of the 27B's linears at decode shape (M=1, f32 IO, n=5):

| projection | shape | tilelang metal | torch-eager mps | ratio |
|---|---|---:|---:|---:|
| qkv | [8192 x 5120] | 21.02 ms | 0.800 ms | 26x |
| o | [5120 x 6144] | 25.07 ms | 0.508 ms | 49x |
| gate_up | [34816 x 5120] | 44.00 ms | 2.905 ms | 15x |
| down | [5120 x 17408] | 70.64 ms | 1.448 ms | 49x |
| **one layer** | | **160.7 ms** | **5.66 ms** | **28x** |

At 64 layers that is **10.3 s/token, 0.097 tok/s** — and that counts linears
only, with no attention, norms, or sampling. torch-eager on the same device and
the same shapes would be 2.76 tok/s.

The gap is the kernels, not the hardware: Metal gets three naive FMA gemms
(`make_gemm_nt_naive`, registry.py:63) because Metal's `T.gemm` rejects global
operands, and runs f32 IO rather than bf16, so it also moves twice the bytes.
1406 of 1969 kernel lines are sm90-only.

**A first measurement of this read 26.0 s/token.** It had resolved to
`target c device cpu` — `TILERL_TARGET` was not set in that shell, so it
measured the CPU target and labelled it Metal. The `assert be.device.type ==
"mps"` in the probe exists because of that.

## The 27B checkpoint cannot be moved here

`/work/Qwen3.8-27B-NVFP4` is **22 GB** on the pod. The Mac was at **6.5 GiB free
of 460 GiB** (99% full) when this was measured. So the end-to-end number is
unmeasurable today for a disk reason, not a memory one — 48 GB of unified memory
would hold the weights fine. The per-layer figures above are measured on real
27B shapes with random weights, which prices the arithmetic without needing the
checkpoint.

## Rule

Three things generalize past Metal:

`is_contiguous()` is not "safe to hand to a kernel". A view can be contiguous
and still start at a non-zero offset, and `.contiguous()` will not fix it
because there is nothing to fix by its own predicate. Where an ABI requires a
zero offset, test `storage_offset()` and copy.

`torch.equal` between two arithmetically-equivalent-but-differently-ordered
computations is a target-portability bug waiting for a second backend. It held
on cpu for as long as cpu was the only target that ran it.

A fix for one target, placed in a helper every target shares, is a cost for
every target until someone counts. "CUDA tolerates this, so the copy is
harmless" was wrong by six copies a tick; the count is what turned a plausible
claim into a gate. Count on the target that pays, not on the one that is
convenient — cpu and metal both said 0 here, and sm90 said 6.
