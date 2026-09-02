"""The sm70 ncols=2 GEMV's shipping contract, checked without a GPU.

`ncols=2` pairs output column j with j + Np/2 so one X load feeds two columns
(HFMA2 per LDG 3.53 -> 6.06, worth 1.82x at M=32,
wins/2026-09-03-ncols2-raises-loads-per-fma.md). Two ways that silently goes wrong,
neither visible in a timing run:

1. A PADDED plane makes the pair partner a pad column, and its garbage lands inside
   the `[:Mr, :N]` slice the dispatch keeps. Wrong numbers, no error.
2. The `Np == N` guard silently falls back to ncols=1 for a shape that pads, so a
   future model shape loses 1.82x with nothing failing.

Both are properties of the SHAPES against the plan's padding, so they are integer
arithmetic — no card, no kernel, runs here.
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest

#: Every fp4 linear N in the shipped 27B, from config.qwen38_27b().
#: gate_up = 2*17408, down/attn-o/gdn-out = 5120 or 6144, qkv = 14336, qkvz = 16384.
SHIPPED_N = (34816, 17408, 16384, 14336, 6144, 5120)


def _round_up(x: int, a: int) -> int:
    return -(-x // a) * a


def test_every_shipped_shape_keeps_ncols2():
    """No shipped N may pad, or it silently drops to the 1x kernel.

    The gemv plan's N tile is 4 (_CUDA_PLAN["linear_fp4","gemv"] = (..., 4, 4), and
    bN = _round_up(min(cap, n), tile)), so Np = _round_up(N, 4). A shape that is not
    a multiple of 4 pads, the dispatch's `Np == N` guard trips, and the fast kernel
    is quietly not used.
    """
    from tilerl_kernels.backend import _NCOLS

    for n in SHIPPED_N:
        np_ = _round_up(n, 4)
        assert np_ == n, f"N={n} pads to {np_}: ncols=2 would fall back to 1"
        assert n % 2 == 0, f"N={n} is odd: ncols=2 cannot pair its columns"
    assert _NCOLS == 2, "the shipped default is ncols=2; TILERL_NCOLS=1 is the A/B arm"


def test_the_padding_guard_is_what_rejects_a_padded_shape():
    """The DISPATCH LINE must fall back to ncols=1 for a padded plane, by keyword.

    Negative control for the guard itself: N=5121 pads to 5124, so column j pairs
    with j + 2562, and columns 5121..5123 are pad. Without the guard their garbage
    lands in Y[:, 2562:] inside the kept slice.

    This reads the guard's own source rather than recomputing its arithmetic. A
    first version of this test did the latter, and deleting the guard from
    backend.py left it PASSING -- a test that reimplements its subject verifies
    only itself (errors/2026-09-02-a-golden-test-proves-only-what-it-exercises).
    Source inspection is the honest option here: exercising the real dispatch needs
    a twiddled fp4 plane on a CUDA device, which this GPU-less gate cannot build.
    """
    import inspect

    from tilerl_kernels.backend import Backend

    src = inspect.getsource(Backend.linear_fp4)
    assert "nc2 = _NCOLS if Np == N and N % 2 == 0 else 1" in src, (
        "the sm70 dispatch must gate ncols on Np == N and even N: a padded plane "
        "pairs a real column with a PAD column, and that garbage lands inside the "
        "[:Mr, :N] slice the dispatch keeps"
    )
    assert "ncols=nc" in src, (
        "ncols must reach the factory BY KEYWORD -- positionally it lands on `abl` "
        "and runs an ablation kernel that returns wrong numbers without raising"
    )

    # And name the columns a padded shape would corrupt, so the mechanism is
    # recorded rather than asserted abstractly.
    n = 5121
    np_ = _round_up(n, 4)
    assert np_ != n, "5121 must pad, or this control proves nothing"
    pad_cols = set(range(n, np_))
    partners = {j + np_ // 2 for j in range(np_ // 2)}
    assert pad_cols & partners, (
        f"pad columns {sorted(pad_cols)} are pair partners; their garbage would "
        f"land in Y[:, {np_ // 2}:{n}] which the [:Mr, :N] slice keeps"
    )


def test_ncols_is_gated_to_the_top_rung():
    """ncols=2 must not reach the M<=8 rungs, or dense decode loses 4.9%.

    The mechanism pays where the GEMV is compute-bound. At M=1 it is bandwidth-bound
    (83% of its byte roofline), so there is no arithmetic to win and only the halved
    grid remains -- and the shipped decode shapes are already at 5-33% of peak.
    Measured 39.1 -> 37.2 tok/s at 4096, uniform ~4.9% at every context, reproduced
    by a second nc2 arm to the decimal
    (errors/2026-09-03-ncols2-cost-5-percent-of-decode.md).

    This is a SILENT failure mode: the wrong rung costs throughput and nothing
    raises, which is why the flip shipped with prefill numbers only.

    The gate keys on the COMPILED RUNG, not the row count, so it exempts M<=8 only:
    the ladder is 1/2/4/8/32, and B*W in 9..31 rounds up to 32 and keeps ncols=2.
    """
    import inspect

    from tilerl_kernels.backend import _NCOLS_MIN_M, Backend, _sm70_chunks

    assert _NCOLS_MIN_M == 32, "the top rung is where the GEMV turns compute-bound"
    src = inspect.getsource(Backend.linear_fp4)
    assert "nc = nc2 if Mk >= _NCOLS_MIN_M else 1" in src, (
        "the per-chunk rung must gate ncols: a decode chunk compiles at Mk=1 and "
        "must get the 1-column kernel"
    )
    # Widths the engine actually submits, INCLUDING ones that are not ladder rungs:
    # the sm70 ladder is 1/2/4/8/32 with no rung between 8 and 32, so B*W in 9..31
    # rounds UP to 32 and keeps ncols=2. A first version of this loop probed only
    # 1/2/4/8/32 -- every ladder-exact width -- and so could not have caught the
    # entry's false claim that "a verify tick takes the 8 rung"
    # (errors/2026-09-03-the-ncols-gate-left-spec-decode-on.md).
    on = {rows: [mk >= _NCOLS_MIN_M for _, _, mk in _sm70_chunks(rows)] for rows in
          (1, 2, 4, 8, 9, 12, 16, 24, 31, 32, 40, 512)}
    for rows in (1, 2, 4, 8):
        assert not any(on[rows]), f"M={rows} is below the top rung: ncols must be off"
    for rows in (9, 12, 16, 24, 31, 32):
        assert all(on[rows]), (
            f"M={rows} rounds up to the 32 rung, so ncols is ON -- a verify tick at "
            f"max_batch*(1+spec_depth) in this range is NOT exempt from the gate"
        )
    # A ragged M keeps ncols on its 32-row chunks and off on the tail rung.
    assert on[40] == [True, False], f"M=40 is 32+8, got {on[40]}"
    assert all(on[512]), "prefill is all 32-row chunks: ncols on for every one"


def test_ncols_factory_rejects_what_it_cannot_serve():
    """ncols=2 without xh, or any value but 1/2, must fail loudly at build time."""
    from tilerl_kernels import kernels_linear

    with pytest.raises(ValueError, match="ncols must be 1 or 2"):
        kernels_linear.make_linear_fp4_gemv_sm70_m("cuda", M=8, xh=True, ncols=3)
    with pytest.raises(ValueError, match="ncols=2 requires xh=True"):
        kernels_linear.make_linear_fp4_gemv_sm70_m("cuda", M=8, xh=False, ncols=2)


def test_the_variant_flags_are_keyword_only():
    """A positional 6th argument must not be able to reach `abl`.

    This is the bug that cost a whole A/B: the dispatch passed ncols positionally
    as `(Mk, 4, True, sh, 0, nc)`, nc landed on `abl`, and both arms ran ablation
    kernels that return wrong numbers -- nc=1 became X_REUSE, whose deleted loads
    read as a 3.8x prefill win, and nc=2 became NO_SCALE, which is nearly free and
    so sat right on the baseline. Nothing raised; only a 4x conflict with a recorded
    number exposed it (errors/2026-09-03-the-ab-measured-abl-not-ncols.md).
    """
    import inspect

    from tilerl_kernels import kernels_linear

    p = inspect.signature(kernels_linear.make_linear_fp4_gemv_sm70_m).parameters
    for name in ("min_blocks", "abl", "ncols"):
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only: a variant flag reachable by position can "
            f"be hit by a caller counting arguments, and an ablation kernel silently "
            f"returns wrong numbers instead of raising"
        )


if __name__ == "__main__":
    test_every_shipped_shape_keeps_ncols2()
    test_the_padding_guard_is_what_rejects_a_padded_shape()
    print("ncols=2 contract holds")
