"""Our TileLang inter-chunk state scan against fla's, same inputs, then timing. fla is the
oracle: already matched to our serial core at 7.3e-03 (bf16 resolution)."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from tilerl_kernels import kernels_gdn
from tilerl_kernels.backend import get_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--block-dv", type=int, default=32)
    args = ap.parse_args()
    b = get_backend()
    assert b.device.type == "cuda", "needs TILERL_TARGET=cuda"
    dev = b.device
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    torch.manual_seed(0)
    B, S, H, DK = 1, args.seq, args.hv, args.dk
    k = torch.nn.functional.normalize(
        torch.randn(B, S, H, DK, device=dev), dim=-1).bfloat16()
    w = (torch.randn(B, S, H, DK, device=dev) * 0.1).bfloat16()
    u = torch.randn(B, S, H, DK, device=dev).bfloat16()
    g = (-torch.rand(B, S, H, device=dev) * 0.5)
    # chunk-local inclusive cumsum, which is what both sides expect
    g = g.view(B, S // args.chunk, args.chunk, H).cumsum(2).view(B, S, H).contiguous()
    st = (torch.randn(B, H, DK, DK, device=dev) * 0.1)

    _, vnew_ref, out_ref = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=st, output_final_state=True,
        chunk_size=args.chunk, save_new_value=True)

    kern = kernels_gdn.make_gdn_state_scan(b.target, block_DV=args.block_dv)
    _, out, vnew = kern(k, w, u, g, st.bfloat16(), args.chunk)

    for name, a, c in (("V_new", vnew.float(), vnew_ref.float()),
                       ("state", out.float(), out_ref.float())):
        rel = (a - c).abs().max().item() / max(c.abs().max().item(), 1e-9)
        print(f"  {name:>6}: rel {rel:.3e}   |max| {c.abs().max().item():.4f}")

    def timed(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        return sum(e.time_range.elapsed_us() for e in prof.events()
                   if e.device_type.name == "CUDA") / args.iters

    t_fla = timed(lambda: chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=st, output_final_state=True,
        chunk_size=args.chunk, save_new_value=True))
    t_ours = timed(lambda: kern(k, w, u, g, st.bfloat16(), args.chunk))
    print(f"\n  fla  {t_fla:7.1f} us/layer   ours {t_ours:7.1f} us/layer"
          f"   {t_fla / t_ours:.2f}x")


if __name__ == "__main__":
    main()
