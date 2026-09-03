"""The gate on storing the fp4 block scale as e4m3 instead of f32.

Measured on the real Qwen3.8-27B-NVFP4 checkpoint (scripts/probe_scale_e4m3.py,
2026-09-03): of 936.6M f32 scale elements, **919.1M (98.14%) belong to 165
tensors whose ``weight_scale`` shipped ``F8_E4M3`` on disk** and round-trip
through e4m3 bit-exactly -- 0 of 919.1M change. Storing those as e4m3 saves
3.677 -> 0.919 GB, **12.60% of the 21.89 GB a decode tick streams**, with no
value change at all.

The other 1.86% is a different population and is NOT takeable: 96 of them are
``in_proj_a``/``in_proj_b``, which ship ``BF16 [48, 5120]`` dense with no
checkpoint fp4, so ``reference.pack_fp4`` synthesises their scale as
``block_max / 6`` -- off the e4m3 grid by construction. Round-tripping those
moves **58 of 1000 MMLU answers, 57 at wide gaps** (measured on card 7 against a
same-arm floor of 0 flips), for 0.24% of tick bytes.

So the shipping change is a CONDITIONAL widening, and the risk moves with it.
The question is no longer "do e4m3 values agree with f32 values" -- for the 165
that is true by construction. It is **"does a kernel handed a mixed-dtype scale
population agree with one handed all-f32"**, because a per-tensor dispatch is
what the loader change creates and a dispatch is what silently gets one tensor
wrong. Every fp4 kernel declares ``Scale: T.Tensor((N, K // block),
"float32")`` (kernels.py:511, kernels_linear.py:168/546/601/876), and
``Backend.linear_fp4`` funnels through one ``scale = self._f32(scale)`` at
backend.py:367 -- that widening is the invariant these tests pin.

Negative control run: replacing that ``_f32`` with a dtype-preserving move fails
two of these three tests. It fails **loudly** -- TileLang rejects the e4m3
operand at the kernel boundary with ``ValueError`` from
``cython_wrapper.pyx:99`` -- so a missing widening cannot silently return wrong
numbers on the CPU cell. That is the good case, and it is why the sm90 arm below
is not optional: it is the target where the scale is consumed by three different
hand-written kernels rather than one.
"""

from __future__ import annotations

import pytest
import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

E4M3 = torch.float8_e4m3fn


def _fp4_weights(n: int, k: int, block: int, seed: int = 0):
    """A packed fp4 weight whose scale is already on the e4m3 grid, as the
    checkpoint's own ``weight_scale`` is for 165 of its 264 fp4 tensors."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    wq, scale = reference.pack_fp4(w, block=block)
    # Put the scale on the grid: this is the checkpoint's state for the 98.14%,
    # not a convenience -- an off-grid scale is the 1.86% this gate rejects.
    return wq, scale.to(E4M3).float()


def test_an_e4m3_scale_reaches_the_fp4_kernel_as_f32():
    """The dispatch invariant: whatever dtype the scale is resident in, the
    kernel is handed f32. Backend._f32 is the one place that holds, so a loader
    that keeps e4m3 cannot change which arithmetic the kernel does."""
    be = get_backend()
    n, k, block = 64, 128, 16
    wq, scale = _fp4_weights(n, k, block)
    x = torch.randn(4, k)

    y_f32 = be.linear_fp4(x, wq, scale)
    y_e4m3 = be.linear_fp4(x, wq, scale.to(E4M3))

    assert y_f32.dtype == y_e4m3.dtype, (y_f32.dtype, y_e4m3.dtype)
    # Bitwise, not allclose: the scale VALUES are identical (grid-aligned), so
    # the only thing under test is that the widening happened at all. Any
    # difference here means the kernel saw two different scale tensors.
    assert torch.equal(y_f32, y_e4m3), (
        f"an e4m3-resident scale changed the output by "
        f"{(y_f32 - y_e4m3).abs().max().item():.3e}: the kernel did not receive f32"
    )


def test_a_mixed_dtype_scale_population_agrees_with_all_f32():
    """The gate the conditional loader actually needs. Half the tensors keep
    e4m3, half stay f32 -- exactly what predicating on ``weight_scale``'s
    presence produces -- and every output must match the all-f32 arm."""
    be = get_backend()
    n, k, block = 32, 64, 16
    x = torch.randn(2, k)

    outs_uniform, outs_mixed = [], []
    for i in range(6):
        wq, scale = _fp4_weights(n, k, block, seed=i)
        outs_uniform.append(be.linear_fp4(x, wq, scale))
        # Odd tensors "shipped a weight_scale" and stay e4m3; even ones don't.
        outs_mixed.append(be.linear_fp4(x, wq, scale.to(E4M3) if i % 2 else scale))

    for i, (a, b) in enumerate(zip(outs_uniform, outs_mixed)):
        assert torch.equal(a, b), (
            f"tensor {i} ({'e4m3' if i % 2 else 'f32'}) diverged by "
            f"{(a - b).abs().max().item():.3e} in a mixed population"
        )


def test_an_off_grid_scale_is_not_bit_exact_through_e4m3():
    """The negative control, and the reason the split is 98.14/1.86 rather than
    100/0. ``pack_fp4``'s raw ``block_max / 6`` is not on the e4m3 grid, so
    round-tripping it DOES change the output -- if this passes silently, the
    test above proves nothing about grid alignment."""
    be = get_backend()
    n, k, block = 32, 64, 16
    g = torch.Generator().manual_seed(7)
    wq, raw = reference.pack_fp4(torch.randn(n, k, generator=g), block=block)
    x = torch.randn(2, k)

    changed = int((raw.to(E4M3).float() != raw).sum())
    assert changed > 0, (
        "a synthesised block_max/6 scale round-tripped e4m3 bit-exactly: this "
        "control cannot distinguish grid-aligned from off-grid, so the "
        "bit-exactness assertions above are vacuous"
    )
    y_raw = be.linear_fp4(x, wq, raw)
    y_rt = be.linear_fp4(x, wq, raw.to(E4M3).float())
    assert not torch.equal(y_raw, y_rt), (
        f"{changed} scale elements changed value yet the output did not: the "
        "kernel is ignoring the scale"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sm90 arm")
def test_the_sm90_fp4_arms_agree_with_the_cpu_reference_on_an_e4m3_scale():
    """pending-remote until run on sm90: the CPU cell and the sm90 cells are
    different kernels (registry.py registers make_linear_fp4 for cpu and
    make_linear_fp4_mma/_gemv/_mma8 for sm90), so CPU parity does not cover
    the arms the 27B actually runs. Same failure shape as the attn_prep hole:
    a bf16 store registered only in _SM90_KERNELS was invisible to every CPU
    gate."""
    be = get_backend()
    n, k, block = 256, 512, 16
    wq, scale = _fp4_weights(n, k, block)
    for m in (1, 2, 8, 64):  # the M values that pick gemv / mma8 / mma
        x = torch.randn(m, k)
        y_f32 = be.linear_fp4(x, wq, scale)
        y_e4m3 = be.linear_fp4(x, wq, scale.to(E4M3))
        assert torch.equal(y_f32, y_e4m3), (
            f"M={m}: e4m3-resident scale moved the sm90 output by "
            f"{(y_f32 - y_e4m3).abs().max().item():.3e}"
        )


if __name__ == "__main__":  # runnable check
    test_an_e4m3_scale_reaches_the_fp4_kernel_as_f32()
    test_a_mixed_dtype_scale_population_agrees_with_all_f32()
    test_an_off_grid_scale_is_not_bit_exact_through_e4m3()
    print("fp4 e4m3 scale parity: ok")
