"""The attention prelude's parity oracle, and why a CPU gate cannot use it.

`reference.attn_prelude` is the f64 norm+rope that both preludes approximate:
sm90's fused `backend.attn_prep`, and the discrete
`rmsnorm`/`rope`/`write_tokens` chain every other cell takes. Measured on the 27B
(card 6), the discrete chain is 2.0007x further from it on 580/580 differing
K-plane elements -- exactly one extra bf16 rounding.
See errors/2026-09-03-unfused-prelude-double-rounds.md.

Three tests, and the middle one is the point: the CPU cell cannot see the defect,
so a CPU parity gate would pass with it present.
"""

import pytest
import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend
from tilerl_kernels.registry import _resolve

from tilerl.config import qwen38_27b


def test_the_oracle_agrees_with_the_reference_rope_it_inlines():
    """`attn_prelude` inlines the rotation at f64 because `_rope_apply` narrows to
    f32. That copy must stay equal to `reference.rope` or it will drift and then
    get quoted as an independent check."""
    cfg = qwen38_27b()
    torch.manual_seed(0)
    x = torch.randn(2, 5, 3, cfg.head_dim, dtype=torch.float32)
    w = torch.randn(cfg.head_dim, dtype=torch.float32)
    pos = torch.arange(5, dtype=torch.int32)
    got = reference.attn_prelude(x, w, pos, cfg.rope_theta, cfg.rms_eps,
                                 rotary_dim=cfg.effective_rotary_dim)
    want = reference.rope(reference.rmsnorm(x, w, cfg.rms_eps), pos, cfg.rope_theta,
                          rotary_dim=cfg.effective_rotary_dim)
    # f32 vs f64 compute of the same formula: agreement to f32 rounding, and the
    # tail past rotary_dim must be untouched by both
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5), (got - want).abs().max()
    rd = cfg.effective_rotary_dim
    assert torch.equal(got[..., rd:], want[..., rd:]), "the pass-through tail diverged"


def test_the_cpu_cell_cannot_observe_the_preludes_extra_rounding():
    """Why the parity gate for this defect has to run on sm90.

    Two facts, both from the registry: `rmsnorm_fused` (whose `Y` is bf16,
    kernels.py:110) is registered only in `_SM90_KERNELS`, and `_CPU_KERNELS`
    maps `rmsnorm_apply` to the f32-output variant. So the CPU chain is f32
    throughout and does not double-round -- a CPU parity test comparing the two
    preludes would pass while sm90 is 2x off.

    This passing is what documents the hole. If it starts failing, the CPU cell
    grew a bf16 norm output and a CPU parity gate became possible."""
    cpu = _resolve("fp4", "cpu")
    assert "rmsnorm_fused" not in cpu, (
        "the CPU cell now has rmsnorm_fused: check its output dtype -- if bf16, a "
        "CPU parity gate for the prelude is now possible and should replace this test"
    )
    assert cpu["rmsnorm_apply"].__name__ == "make_rmsnorm_apply", (
        f"CPU rmsnorm_apply is now {cpu['rmsnorm_apply'].__name__}: if it emits bf16, "
        "the CPU cell double-rounds too and this test's premise is gone"
    )
    sm90 = _resolve("fp4", "sm90")
    assert sm90["rmsnorm_apply"].__name__ == "make_rmsnorm_apply_bf16"
    assert "rmsnorm_fused" in sm90
    # no fallback route avoids it: Backend.rmsnorm takes rmsnorm_fused when present
    # and otherwise rmsnorm_partial + rmsnorm_apply, and sm90 overrides both
    assert sm90["rmsnorm_fused"].__name__ == "make_rmsnorm_fused_bf16"


@pytest.mark.skipif(get_backend().arch != "sm90", reason="attn_prep is sm90-only")
def test_attn_prep_is_closer_to_exact_than_the_discrete_prelude():
    """The gate the fix needs, on the target that can see it.

    Ranked by mean error over the elements that actually differ, never by max:
    both arms round to the same bf16 grid, so max|d| is the quantum at the same
    largest element in both and reads as a tie to four figures."""
    be = get_backend()
    cfg = qwen38_27b()
    d, rd = cfg.head_dim, cfg.effective_rotary_dim
    torch.manual_seed(0)
    x = torch.randn(2, 8, cfg.num_kv_heads, d, dtype=torch.float32, device=be.device)
    w = torch.randn(d, dtype=torch.float32, device=be.device)
    pos = torch.arange(8, dtype=torch.int32, device=be.device).unsqueeze(0).expand(2, -1)

    ref = reference.attn_prelude(x, w, pos, cfg.rope_theta, cfg.rms_eps, rotary_dim=rd)
    # the discrete chain, exactly as model.py:239-242 calls it, then the pool's cast
    disc = be.rope(be.rmsnorm(x, w, cfg.rms_eps), pos, cfg.rope_theta, rotary_dim=rd)
    disc = disc.to(torch.bfloat16).float()

    ed = (disc - ref).abs()
    assert ed.max().item() > 0, "the discrete chain matched f64 exactly: check the harness"
    # one bf16 rounding of a value already rounded once is ~2x the error of one
    print(f"discrete vs f64 oracle: mean {ed.mean().item():.3e} max {ed.max().item():.3e}")
