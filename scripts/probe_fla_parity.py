"""fla's chunked core vs our serial one, same inputs. fla runs in bf16, so the bar is bf16
rounding, not exactness."""

from __future__ import annotations

import argparse

import torch
from tilerl_kernels import reference as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--hk", type=int, default=16)
    ap.add_argument("--hv", type=int, default=48)
    ap.add_argument("--dk", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    b, t = 1, args.seq
    qn = torch.randn(b, t, args.hk, args.dk, device=dev)
    qn = qn / qn.norm(dim=-1, keepdim=True) / args.dk ** 0.5
    kn = torch.randn(b, t, args.hk, args.dk, device=dev)
    kn = kn / kn.norm(dim=-1, keepdim=True)
    v = torch.randn(b, t, args.hv, args.dk, device=dev)
    gt = -torch.rand(b, t, args.hv, device=dev) * 0.5
    bt = torch.rand(b, t, args.hv, device=dev)
    st = torch.randn(b, args.hv, args.dk, args.dk, device=dev) * 0.1

    ours = R.gdn_chunk_core(qn, kn, v, gt, bt, st, chunk=args.chunk)
    theirs = R.gdn_chunk_core_fla(qn, kn, v, gt, bt, st, chunk=args.chunk)
    for name, a, c in (("core", ours[0], theirs[0]), ("state", ours[1], theirs[1])):
        rel = (a - c).abs().max().item() / max(a.abs().max().item(), 1e-9)
        print(f"  {name:>6}: rel {rel:.3e}   ours |max| {a.abs().max():.4f}")


if __name__ == "__main__":
    main()
