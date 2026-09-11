"""sm70 (V100) sparse-KV scorer parity against the CPU reference.

Runs on the V100 via scripts/v100.sh:
  TILERL_TARGET=cuda PYTHONPATH=src:packages/tilerl-kernels/src \
    /usr/bin/python3 scripts/probe_sparse_sm70.py

page_bounds: sm70 stores f16 bounds, so parity is f16-tight (elementwise,
rtol/atol ~1e-3 relative). page_bound_scores accumulates f32 from the f16
bounds, so compare to the reference run on the SAME f16-rounded bounds.
"""

import torch
from tilerl_kernels import reference
from tilerl_kernels.backend import get_backend


def main():
    be = get_backend()
    print("arch", be.arch, "device", torch.cuda.get_device_name(0))
    assert be.arch == "sm70", "this probe is the sm70 cell"
    torch.manual_seed(0)

    P, H, BLK, D, Tq = 8, 4, 16, 256, 1
    k = torch.randn(P, H, BLK, D, dtype=torch.float32) * 0.3
    q = torch.randn(Tq, H, D, dtype=torch.float32)

    # bounds: card kernel narrows to f16; reference in f32, then compare the
    # card output against the reference computed on f16-rounded bounds.
    bnd = be.page_bounds(k.cuda())
    print("bounds dtype/shape:", bnd.dtype, tuple(bnd.shape))
    assert bnd.dtype == torch.float16, bnd.dtype
    ref_bnd_full = reference.page_bounds(k)
    ref_bnd = ref_bnd_full.half()
    err_b = (bnd.float().cpu() - ref_bnd.float()).abs().max().item()
    print("bounds max abs err vs f16 reference:", err_b)
    assert torch.allclose(bnd.float().cpu(), ref_bnd.float(), atol=1e-2, rtol=1e-2)

    # scores: f32 accumulator, fed the SAME f16 bounds on both sides
    sc = be.page_bound_scores(q.cuda(), bnd).cpu()
    ref_sc = reference.page_bound_scores(q, ref_bnd.half().float())
    err_s = (sc - ref_sc).abs().max().item()
    rel = err_s / ref_sc.abs().max().item()
    print("scores max abs err", err_s, "max rel", rel)
    assert torch.allclose(sc, ref_sc, atol=1e-2, rtol=2e-2), (err_s, rel)

    # select_pages is a torch gather (no arch kernel yet) — sanity on cuda input
    bt = (torch.arange(P) + 1).reshape(1, P).cuda()
    scores = sc.sum(1).reshape(1, 1, P).cuda()
    sel = reference.select_pages(bt, torch.tensor([P]).cuda(), scores, P // 2)
    print("select_pages k=P/2:", sel[0, 0].tolist())
    assert sel.shape == (1, 1, P // 2)
    # sequence-order: chosen indices ascending
    pos = (sel[0, 0] - 1).cpu().tolist()
    assert pos == sorted(pos), pos

    print("PROBE_OK")


if __name__ == "__main__":
    main()
