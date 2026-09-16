"""Parity for the Quest sparse-KV bound kernels (audit F22), runnable in CI.

`page_bounds` (per page/KV-head min,max of K over its 16 tokens) and
`page_bound_scores` (the Quest upper-bound score per page) had a registered
TileLang kernel ONLY in the sm70 cell, and every existing test called the torch
reference directly -- so the kernel had no parity gate on any arch.

Key fact (why this runs on CPU, unlike the F6 prelude gate): both kernels are
target-INDEPENDENT -- pure f32 elementwise min/max and a dot product, no arch
intrinsic. `kernels.make_page_bounds("c")` / `make_page_bound_scores("c")`
JIT-compile and run under the C backend on a CPU CI host, so the correctness gate
runs every push. We call the FACTORY directly, not `backend.page_bounds`: the
backend silently returns the reference when a cell lacks the key, which would
make a backend-routed CPU test compare the reference to itself -- exactly the
"only tests the reference" hole F22 names.

* bounds: the min/max over 16 has no arithmetic accumulation, so the f32 kernel
  is BIT-EXACT vs reference (asserted exact, no tolerance).
* scores: a serial f32 sum over D vs torch's broadcast sum differs only by
  reduction order; max relative error ~1e-7 measured, gate rtol 1e-4 (~100x
  margin, 100x tighter than the project's bf16 1e-2, tight enough that dropping
  a dimension, swapping kmin/kmax, or using only one max-arm fails O(1)).

The sm90-specific work is WIRING only (the kernel is the same target-independent
function): the sm90 cell registers both keys so the backend no longer falls
back. That is a pure registry test (runs everywhere) plus an on-sm90 smoke that
the backend actually dispatches the kernel. Verified on H20; the sm70 f16 cell
pre-dates this and keeps its narrower storage.
"""

from __future__ import annotations

import pytest
import torch
from tilerl_kernels import kernels
from tilerl_kernels.backend import get_backend
from tilerl_kernels.reference import page_bound_scores, page_bounds
from tilerl_kernels.registry import _resolve

#: reduction-order band for page_bound_scores, measured ~1e-7, headroom to ~1e-4.
SCORE_RTOL = 1e-4
SCORE_ATOL = 1e-3
_THREADS = 128


def _case(*, p=64, hkv=8, blk=16, d=128, tq=1):
    """K spans a wide per-token magnitude range so kmin != kmax and scores are
    large/nondegenerate (D=128 = the 27B linear_key_head_dim)."""
    torch.manual_seed(0)
    k = torch.randn(p, hkv, blk, d, dtype=torch.float32)
    k = k * torch.logspace(-1, 1, blk, dtype=torch.float32).view(1, 1, blk, 1)
    q = torch.randn(tq, hkv, d, dtype=torch.float32)
    return k, q


def _kernels(device):
    kb = kernels.make_page_bounds("c", out_dtype="float32")
    ks = kernels.make_page_bound_scores("c")
    return (
        lambda k: kb(k.contiguous(), threads=_THREADS),
        lambda q, b: ks(q.contiguous(), b.contiguous(), threads=_THREADS),
    )


def test_page_bounds_is_bit_exact_to_reference():
    k, _ = _case()
    got, _ = _kernels(k.device)
    got = got(k)
    want = page_bounds(k)
    assert torch.any(got[..., 0, :] != got[..., 1, :]), (
        "kmin == kmax everywhere: input does not exercise the min/max reduction")
    assert got.dtype == torch.float32
    # min/max of 16 f32 values in any order is the same set of values -> exact.
    assert torch.equal(got, want), (
        f"f32 page_bounds not bit-exact: maxdiff {(got - want).abs().max().item()}")


def test_page_bound_scores_matches_reference_within_reduction_order():
    k, q = _case()
    kb, ks = _kernels(k.device)
    bounds = kb(k)
    got, want = ks(q, bounds), page_bound_scores(q, bounds)

    assert got.abs().max().item() > 1.0, "scores ~zero: the comparison would be vacuous"
    rel = ((got - want).abs() / want.abs().clamp_min(1e-6)).max().item()
    assert torch.allclose(got, want, rtol=SCORE_RTOL, atol=SCORE_ATOL), (
        f"page_bound_scores kernel vs reference: maxrel {rel:.2e} exceeds "
        f"rtol {SCORE_RTOL} (reduction order is the only legal difference)")


@pytest.mark.parametrize("mistake", ["half_dim", "kmin_only"])
def test_score_band_rejects_structural_mistakes(mistake):
    """Red-control: the rtol band must reject O(1) structural errors, not only
    reduction-order noise. half_dim drops half the head dim; kmin_only scores
    q*kmin without the max(q*kmin, q*kmax). Both are legal f32 sums, so the
    test proves the band discriminates structure, not dtype noise."""
    k, q = _case()
    d = k.shape[-1]
    kmin, kmax = page_bounds(k).unbind(dim=2)  # [p,h,d]
    q0 = q.unsqueeze(0)  # [1,tq,h,d]
    if mistake == "half_dim":
        wrong = (q0[..., : d // 2] * kmax[..., : d // 2].unsqueeze(1)).sum(dim=(1, 3))
    else:
        wrong = (q0 * kmin.unsqueeze(1)).sum(dim=(1, 3))
    good = page_bound_scores(q, page_bounds(k))
    assert not torch.allclose(wrong, good, rtol=SCORE_RTOL, atol=SCORE_ATOL), (
        f"{mistake}: an O(1) structural error slipped through the score band -- "
        "tolerance is too loose")


def test_sm90_cell_registers_quest_kernels():
    """F22 structural gate: sm90 must register both kernels, not silently fall
    back to reference. Pure registry lookup, runs on every arch (incl CI)."""
    cell = _resolve("fp4", "sm90")
    assert "page_bounds" in cell, "sm90 missing page_bounds (silent reference fallback)"
    assert "page_bound_scores" in cell, "sm90 missing page_bound_scores"


@pytest.mark.skipif(get_backend().arch != "sm90", reason="dispatch wiring is sm90-only")
def test_sm90_backend_dispatches_the_kernel_not_the_reference():
    """On sm90 the backend route must resolve to the registered kernel. Both
    the key being present and the output matching the (target-independent)
    kernel confirm wiring end to end on the device."""
    be = get_backend()
    assert "page_bounds" in _resolve(be.precision, be.arch)
    assert "page_bound_scores" in _resolve(be.precision, be.arch)
    k, q = _case()
    bounds = be.page_bounds(k)
    assert bounds.dtype == torch.float32, "sm90 bounds store f32"
    scores = be.page_bound_scores(q, bounds)
    assert torch.allclose(scores, page_bound_scores(q, bounds),
                          rtol=SCORE_RTOL, atol=SCORE_ATOL)


def test_kernel_bounds_rank_pages_the_same_as_reference():
    """Behavioral invariant (runs in CI via the C-target kernel): the page
    ranking derived from kernel bounds must equal that from reference bounds.
    The bounds dtype/impl (f32 here; f16 on sm70, widened by the scorer) must
    not change which pages Quest would attend. Compare the top-k index set of
    per-page scores directly, bypassing the layer-batched select_pages."""
    k, q = _case(p=64)
    p = k.shape[0]
    kb, _ = _kernels(k.device)
    topk = p // 2

    def page_rank(bounds):
        return torch.topk(page_bound_scores(q, bounds).amax(dim=1), topk).indices.sort().values

    kernel_rank = page_rank(kb(k))
    ref_rank = page_rank(page_bounds(k))
    assert torch.equal(kernel_rank, ref_rank), (
        f"bounds impl changed the top-{topk} page set: "
        f"{kernel_rank.tolist()} vs {ref_rank.tolist()}")


if __name__ == "__main__":
    cell = _resolve("fp4", "sm90")
    print("sm90 quest registered:",
          "page_bounds" in cell and "page_bound_scores" in cell)
