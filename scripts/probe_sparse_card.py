"""Card sparse-KV scorer parity against the CPU reference (sm70 or sm90).

sm70 stores f16 bounds; sm90 stores bf16 (its IO dtype). The score
accumulates f32 from those rounded bounds on both arches. Run on a card:

  TILERL_TARGET=cuda PYTHONPATH=src:packages/tilerl-kernels/src \
    /usr/bin/python3 scripts/probe_sparse_card.py
"""

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend

BOUNDS_DTYPE = {"sm70": torch.float16, "sm90": torch.bfloat16}


def main():
    be = get_backend()
    print("arch", be.arch, "device", torch.cuda.get_device_name(0))
    assert be.arch in BOUNDS_DTYPE, f"no sparse bounds cell for {be.arch}"
    torch.manual_seed(0)

    P, H, BLK, D, Tq = 8, 4, 16, 256, 1
    k = torch.randn(P, H, BLK, D, dtype=torch.float32) * 0.3
    q = torch.randn(Tq, H, D, dtype=torch.float32)
    bdtype = BOUNDS_DTYPE[be.arch]

    bnd = be.page_bounds(k.cuda())
    print("bounds dtype/shape:", bnd.dtype, tuple(bnd.shape))
    assert bnd.dtype == bdtype, bnd.dtype
    ref_bnd = reference.page_bounds(k).to(bdtype)
    err_b = (bnd.float().cpu() - ref_bnd.float()).abs().max().item()
    print("bounds max abs err vs", bdtype, "reference:", err_b)
    assert torch.allclose(bnd.float().cpu(), ref_bnd.float(), atol=1e-2, rtol=1e-2)

    sc = be.page_bound_scores(q.cuda(), bnd).cpu()
    ref_sc = reference.page_bound_scores(q, ref_bnd.float())
    err_s = (sc - ref_sc).abs().max().item()
    rel = err_s / ref_sc.abs().max().item()
    print("scores max abs err", err_s, "max rel", rel)
    assert torch.allclose(sc, ref_sc, atol=2e-2, rtol=2e-2), (err_s, rel)

    bt = (torch.arange(P) + 1).reshape(1, P).cuda()
    scores = sc.sum(1).reshape(1, 1, P).cuda()
    sel = reference.select_pages(bt, torch.tensor([P]).cuda(), scores, P // 2)
    print("select_pages k=P/2:", sel[0, 0].tolist())
    assert sel.shape == (1, 1, P // 2)
    pos = (sel[0, 0] - 1).cpu().tolist()
    assert pos == sorted(pos), pos

    print(f"PROBE_OK {be.arch}")


if __name__ == "__main__":
    main()
