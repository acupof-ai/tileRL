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
    """The gate F6 needs, on the only target that can see it: the FUSED prelude
    must land closer to the f64 oracle than the discrete rmsnorm->rope chain.

    The discrete chain rounds its norm output to bf16 before RoPE; the fused
    ``attn_prep`` keeps norm+RoPE in f32 registers and casts once at the store, so
    on the elements where the two differ the fused mean error is about half the
    discrete one (measured 2.0007x on the 27B: errors/2026-09-03
    -unfused-prelude-double-rounds.md). Both arms write into a real PagedKvPool
    because ``attn_prep`` performs the K/V write itself; only q is compared here
    (q carries the two-prelude rounding difference with no pool dtype gap).

    Ranked by mean error over the elements that ACTUALLY differ, never by max:
    both arms round to the same bf16 grid, so max|d| is the quantum at the same
    largest element in both and reads as a tie. The fused==discrete case is the
    negative control: then ef_mean == ed_mean and the strict ``<`` fails.
    """
    from tilerl.kv_cache import PagedKvPool

    be = get_backend()
    cfg = qwen38_27b()
    d, rd = cfg.head_dim, cfg.effective_rotary_dim
    hq, hkv = cfg.num_attention_heads, cfg.num_kv_heads
    b, s = 2, 8
    # Full gated layout: [query; gate] per head, then k, v -- exactly what attn_prep
    # reads (it normalizes the query half of the interleaved 2*D block).
    nqkv = hq * d * (2 if cfg.full_attn_gated else 1) + 2 * hkv * d
    torch.manual_seed(0)
    qkv = torch.randn(b, s, nqkv, dtype=torch.float32, device=be.device)
    pos = torch.arange(s, dtype=torch.int32, device=be.device).unsqueeze(0).expand(b, -1)
    # Random norm weights are enough to expose the rounding-direction difference;
    # the gate does not need the 27B checkpoint.
    wq = torch.randn(d, dtype=torch.float32, device=be.device)
    wk = torch.randn(d, dtype=torch.float32, device=be.device)

    # --- fused arm: one launch does q_norm + rope (and the K/V write) ---
    fused_pool = PagedKvPool(b * 8, hkv, d, num_layers=1, device=be.device)
    fused_kv = _PreludeKv(fused_pool, b, s, be.device)
    q_fused = be.attn_prep(qkv, wq, wk, pos, cfg.rope_theta, rd, fused_kv,
                           0, hq, hkv, cfg.rms_eps)
    assert q_fused is not None, "no fused attn_prep kernel registered in this sm90 cell"

    # --- discrete arm: reshape, rmsnorm, rope as model.py does it ---
    q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
    q = qkv[..., :q_rows]
    q = q.reshape(b, s, hq, 2, d)[..., 0, :] if cfg.full_attn_gated \
        else q.reshape(b, s, hq, d)
    q_disc = be.rope(be.rmsnorm(q, wq, cfg.rms_eps), pos, cfg.rope_theta,
                     rotary_dim=rd)
    q_disc = q_disc.to(torch.bfloat16).float()  # the pool's one cast, as in serving

    # --- f64 oracle both approximate ---
    q_ref = reference.attn_prelude(q, wq, pos, cfg.rope_theta, cfg.rms_eps,
                                   rotary_dim=rd).float()

    ed = (q_disc - q_ref).abs()
    ef = (q_fused.float() - q_ref).abs()
    differ = ed != ef
    # Non-triviality: if the discrete chain matched f64 exactly there would be no
    # double rounding to rank -- keeps this from passing on a vacuous comparison.
    assert ed.max().item() > 0, "the discrete chain matched f64 exactly: check the harness"
    assert differ.any(), (
        "fused and discrete are bit-identical on q: no prelude-rounding difference "
        "is visible in this shape -- the closer-than comparison would be vacuous")

    ed_mean = ed[differ].mean().item()
    ef_mean = ef[differ].mean().item()
    print(f"on {int(differ.sum())} differing elements: discrete mean {ed_mean:.3e}, "
          f"fused mean {ef_mean:.3e}, ratio {ed_mean / max(ef_mean, 1e-30):.4f}")
    # Fused keeps one less bf16 rounding, so it must be strictly closer. The 0.6
    # band codifies the ~half (measured 2.0007x -> fused ~0.4996) and still fails
    # on fused==discrete (ratio 1.0) or a fused regression above 0.6.
    assert ef_mean < ed_mean, (
        f"fused prelude is not closer to exact: fused {ef_mean:.3e} >= "
        f"discrete {ed_mean:.3e} -- the extra bf16 rounding is not being avoided")
    assert ef_mean <= 0.6 * ed_mean, (
        f"fused mean {ef_mean:.3e} is not ~half the discrete {ed_mean:.3e} "
        f"(ratio {ef_mean / ed_mean:.3f}); expected one fewer bf16 rounding, ~0.5x")


class _PreludeKv:
    """Minimal batch state attn_prep reads off a KV pool (mirrors
    scripts/probe_attn_prep.py): contiguous block_table with s fitting block 0."""

    dense = False

    def __init__(self, pool, b, s, device):
        self.kv_pool = pool
        nb = pool.k_pool.shape[-2]
        assert s <= nb, f"s={s} must fit one block ({nb})"
        self.block_table = torch.arange(b * 8, dtype=torch.int32,
                                        device=device).reshape(b, 8)
        self.seq_len = torch.full((b,), s, dtype=torch.int32, device=device)
        self.seq_q_lens = torch.full((b,), s, dtype=torch.int32, device=device)
