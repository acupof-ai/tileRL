"""The six-kernel TileLang WY path (kernels_gdn) against fla, kernel by kernel and end to
end, at our shapes; then the known-answer rows for the state scan, then timing.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:packages/tilerl-kernels/src \\
    TILERL_TARGET=cuda python3 scripts/probe_gdn_wy.py
"""

from __future__ import annotations

import argparse

import torch
from tilerl_kernels import reference as R
from tilerl_kernels.backend import get_backend
from torch.profiler import ProfilerActivity, profile


def err(name, ours, ref):
    ours, ref = ours.float(), ref.float()
    mx = (ours - ref).abs().max().item()
    rel = mx / max(ref.abs().max().item(), 1e-9)
    print(f"  {name:>12}: max abs {mx:.3e}   rel {rel:.3e}   |ref| {ref.abs().max():.4f}")
    return rel


def timed(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    by: dict[str, float] = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            by[e.name[:46]] = by.get(e.name[:46], 0.0) + e.time_range.elapsed_us() / iters
    return by


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hk", type=int, default=16)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--dv", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    bk = get_backend()
    assert bk.device.type == "cuda", "needs TILERL_TARGET=cuda"
    dev = bk.device
    # fla 0.5.2 signatures; its solve_tril module attr is shadowed by the function
    from importlib import import_module

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.utils.cumsum import chunk_local_cumsum

    solve_tril = import_module("fla.ops.utils.solve_tril").solve_tril
    try:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_intra
    except ImportError:
        chunk_gated_delta_rule_fwd_intra = None
    RCP_LN2 = 1.4426950216  # fla's forward keeps G in log2 units and uses exp2; ours is ln

    torch.manual_seed(0)
    b, t, C = 1, args.seq, args.chunk
    qn = torch.randn(b, t, args.hk, args.dk, device=dev)
    qn = qn / qn.norm(dim=-1, keepdim=True) / args.dk**0.5
    kn = torch.randn(b, t, args.hk, args.dk, device=dev)
    kn = kn / kn.norm(dim=-1, keepdim=True)
    v = torch.randn(b, t, args.hv, args.dv, device=dev)
    gt = -torch.rand(b, t, args.hv, device=dev) * 0.5
    bt = torch.rand(b, t, args.hv, device=dev)
    st = torch.randn(b, args.hv, args.dk, args.dv, device=dev) * 0.1

    # (a) each kernel on fla's own inputs for that stage (fla in log2 gate units, ours in ln)
    rep = args.hv // args.hk
    q = qn.repeat_interleave(rep, dim=2).bfloat16()
    k = kn.repeat_interleave(rep, dim=2).bfloat16()
    vb, beta = v.bfloat16(), bt.bfloat16()
    kern = bk._kernel
    print("(a) per kernel vs fla")
    g_ref = chunk_local_cumsum(gt, C)  # natural units, what our kernels consume
    g2 = chunk_local_cumsum(gt, C, scale=RCP_LN2)  # log2 units, what fla's kernels consume
    err("cumsum", kern("gdn_chunk_cumsum")(gt, C), g_ref)
    a_ref = chunk_scaled_dot_kkt_fwd(k, g2, beta, chunk_size=C, output_dtype=torch.float32)
    err("kkt", kern("gdn_chunk_kkt")(k, beta, g_ref, C), a_ref)
    ai_ref = solve_tril(a_ref, output_dtype=torch.bfloat16)
    err("solve_tril", kern("gdn_solve_tril")(a_ref, C), ai_ref)
    w_ref, u_ref = recompute_w_u_fwd(k, vb, beta, ai_ref, g2)
    w, u = kern("gdn_chunk_wu")(k, vb, beta, g_ref, ai_ref, C)
    err("w", w, w_ref)
    err("u", u, u_ref)
    if chunk_gated_delta_rule_fwd_intra is not None:  # fla's fused kkt+solve+wu
        w_i, u_i, _ = chunk_gated_delta_rule_fwd_intra(k, vb, g2, beta, chunk_size=C)
        err("w (intra)", w, w_i)
        err("u (intra)", u, u_i)
    h_ref, vnew_ref, s_ref = chunk_gated_delta_rule_fwd_h(
        k,
        w_ref,
        u_ref,
        g2,
        initial_state=st,
        output_final_state=True,
        chunk_size=C,
        save_new_value=True,
    )
    h, s, vnew = kern("gdn_state_scan")(k, w_ref, u_ref, g_ref, st.bfloat16(), C)
    # state_v_first=False: fla's h/final_state are [.., DK, DV] like ours; the transposed
    # row is printed alongside so a layout mismatch reads as such, not as arithmetic
    print(f"  fla h {tuple(h_ref.shape)} ours {tuple(h.shape)}")
    err("h (chunks)", h, h_ref)
    err("h^T", h, h_ref.transpose(-1, -2))
    err("V_new", vnew, vnew_ref)
    err("final state", s, s_ref)
    err("final^T", s, s_ref.transpose(-1, -2))
    o_ref = chunk_fwd_o(q, k, vnew_ref, h_ref, g2, scale=1.0, chunk_size=C)
    err("o", kern("gdn_chunk_o")(q, k, vnew_ref, h_ref, g_ref, C, 1.0), o_ref)

    # (b) end to end: vs the serial-chunk reference, then vs fla's full chunk_gated_delta_rule
    print("(b) Backend._gdn_wy_core vs reference.gdn_chunk_core / gdn_chunk_core_fla")
    # the unrepeated key heads, the shape gdn_prep emits: the kernels index the GQA group
    qh, kh_ = bk._c(qn.bfloat16()), bk._c(kn.bfloat16())
    wy_core = lambda: bk._gdn_wy_core(qh, kh_, vb, gt, beta, st, C)
    core, s = wy_core()
    core_ref, s_ref = R.gdn_chunk_core(qn, kn, v, gt, bt, st, chunk=C)
    worst = max(err("core", core, core_ref), err("state", s, s_ref))
    core_fla, s_fla = R.gdn_chunk_core_fla(qn, kn, v, gt, bt, st, chunk=C)
    err("core (fla)", core, core_fla)
    err("state (fla)", s, s_fla)
    assert worst <= 1e-2, f"end-to-end rel {worst:.3e} > 1e-2"

    # (b1)/(b2) the sm90 gdn_prep and gdn_post cells: they only ever run here and on the
    # pod harness, never under the CPU parity gate. Amplitude 1.0, the scale
    # test_gdn_chunk_fused_parity_full_scale settled on -- a pipeline that passed at 0.1
    # was 26% wrong at 1.0 (errors/2026-08-25-gdn-chunked-gdr-rejected.md).
    qkvd = 2 * args.hk * args.dk + args.hv * args.dv
    raw = lambda n: torch.randn(b, t, n, device=dev, dtype=torch.bfloat16)
    lw = dict(
        conv1d_weight=torch.randn(qkvd, 4, device=dev) * 0.1,
        dt_bias=torch.randn(args.hv, device=dev),
        a_log=torch.randn(args.hv, device=dev) * 0.1,
        norm_weight=torch.ones(args.dv, device=dev),
        conv_window=torch.randn(b, 3, qkvd, device=dev),
    )
    lq, lk, lv, lz = (raw(args.hk * args.dk), raw(args.hk * args.dk),
                      raw(args.hv * args.dv), raw(args.hv * args.dv))
    lg, lbeta = raw(args.hv), raw(args.hv)

    # (b1) gdn_prep alone. The layer number below cannot separate this cell from the WY
    # six, and a 1e-3 error in it reads as bf16 noise there.
    print("(b1) gdn_prep vs reference.gdn_prep")
    pw = {n: lw[n] for n in ("conv1d_weight", "dt_bias", "a_log", "conv_window")}
    worst = max(
        err(n, a, r)
        for a, r, n in zip(
            bk._gdn_prep(lq, lk, lv, lg, lbeta, st, **pw),
            R.gdn_prep(lq, lk, lv, lg, lbeta, args.dk, **pw),
            ("qn", "kn", "v", "g", "beta", "window"),
        )
    )
    assert worst <= 1e-2, f"gdn_prep rel {worst:.3e} > 1e-2"

    print("(b2) backend.linear_attn_chunk vs reference.gdn_forward")
    got = bk.linear_attn_chunk(lq, lk, lv, lg, lbeta, st, z=lz, **lw)
    ref = R.gdn_forward(lq, lk, lv, lg, lbeta, st, z=lz, **lw)
    worst = max(err(n, a, r) for a, r, n in zip(got, ref, ("out", "state", "window")))
    assert worst <= 1e-2, f"layer rel {worst:.3e} > 1e-2"

    # (c) known-answer rows (one chunk, no cross-chunk carry): W=0 -> V_new = U, S' = e_last S
    print("(c) state-scan known answers")
    z = lambda *sh, dt=torch.bfloat16: torch.zeros(*sh, device=dev, dtype=dt)
    ones = lambda *sh, dt=torch.bfloat16: torch.ones(*sh, device=dev, dtype=dt)
    k0, w0, u0 = z(1, C, 1, args.dk), z(1, C, 1, args.dk), z(1, C, 1, args.dv)
    g0, s0 = z(1, C, 1, dt=torch.float32), z(1, 1, args.dk, args.dv)
    scan = lambda kk, ww, uu, gg, ss: kern("gdn_state_scan")(kk, ww, uu, gg, ss, C)
    _, out, vn = scan(k0, w0, u0, g0, s0)
    print(
        f"  all zero      -> |vnew| {vn.float().abs().max():.3f} |state| {out.abs().max():.3f}"
        "   (0, 0)"
    )
    u1, s1 = ones(1, C, 1, args.dv), ones(1, 1, args.dk, args.dv)
    _, out, vn = scan(k0, w0, u1, g0, s1)
    print(
        f"  W=K=0,U=1,S=1 -> vnew[0,:2] {vn[0, 0, 0, :2].float().tolist()} "
        f"state[0,:2] {out[0, 0, 0, :2].tolist()}   (1, 1)"
    )
    _, out, _ = scan(ones(1, C, 1, args.dk), w0, u1, g0, s1)
    print(f"  + K=1         -> state[0,:2] {out[0, 0, 0, :2].tolist()}   ({1 + C})")
    for gv in (-0.25, -0.5):
        gg = torch.full((1, C, 1), gv, device=dev).cumsum(1)
        _, out, _ = scan(k0, w0, u0, gg, s1)
        print(
            f"  K=U=0, g={gv:+.2f} -> state[0,0] {out[0, 0, 0, 0]:.4g}"
            f"   ({gg[0, -1, 0].exp():.4g})"
        )

    # (d) timing: the six-kernel path vs fla's chunk_gated_delta_rule
    print("(d) us/layer")
    ours = timed(wy_core, args.iters)
    for name, us in sorted(ours.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<46} {us:>8.1f}")
    fla = timed(
        lambda: chunk_gated_delta_rule(
            q=q,
            k=k,
            v=vb,
            g=gt,
            beta=beta,
            scale=1.0,
            initial_state=st,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        ),
        args.iters,
    )
    print(
        f"  ours {sum(ours.values()):7.1f} us/layer   fla {sum(fla.values()):7.1f}"
        f"   (fla measured 140.7 on 2026-08-29)"
    )


if __name__ == "__main__":
    main()
