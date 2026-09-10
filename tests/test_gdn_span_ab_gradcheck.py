"""GDN span (A, B) transfer: NUMERIC GRADCHECK for the span backward.

``gdn_span_ab`` maps a chunk span's prepped q/k/v/beta/gate to its affine transfer
``s_out = A s_in + B``. Context-parallel training starts each chunk from a scanned
prefix, so the tape needs the cotangent of (A, B) back on the span's inputs — the
op had no reverse, so under CP that gradient was silently absent.

The oracle is independent code: central differences on the scalar
``gA . A + gB . B`` for each prepped input. q does not enter the transfer (only the
token output reads q), so its cotangent is exactly zero.

Control ``--decay-a`` runs a reverse that treats ``a_i`` as the decay scalar
``exp(glast) I`` and drops the ``-R^T W`` operator — the same wrong model the scan
gate uses; it must be caught red.

    TILERL_TARGET=cpu python3 tests/gdn_span_ab_gradcheck.py            # the gate
    TILERL_TARGET=cpu python3 tests/gdn_span_ab_gradcheck.py --decay-a  # red
"""

from __future__ import annotations

import sys

import torch

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

B, T, HV, DK, DV, CHUNK = 1, 6, 2, 6, 4, 2
STEP = 1e-3
TOL = 5e-3
NSAMP = 120


def _inputs():
    torch.manual_seed(11)
    qn = torch.randn(B, T, HV, DK) * 0.3
    kn = torch.randn(B, T, HV, DK) * 0.3
    kn = kn / kn.norm(dim=-1, keepdim=True)
    vn = torch.randn(B, T, HV, DV) * 0.3
    bt = torch.sigmoid(torch.randn(B, T, HV))
    gt = -torch.rand(B, T, HV) * 0.2
    return qn, kn, vn, bt, gt


def _span(qn, kn, vn, bt, gt):
    from tilerl_kernels import reference
    return reference.gdn_span_ab(qn, kn, vn, bt, gt, chunk=CHUNK)


def _decay_span_bwd(ga, gb, qn, kn, vn, bt, gt):
    """Same chain, but a_i is wrongly the decay scalar*I (the -R^T W term dropped).
    Reimplemented minimally so the control corrupts ONLY the reverse's model."""
    from tilerl_kernels import reference
    b, t, hv, dk = qn.shape
    dv, dev, dt = vn.shape[-1], qn.device, qn.dtype
    eye = torch.eye(dk, dtype=dt, device=dev).expand(b, hv, dk, dk).contiguous()
    caches, ais, pref = [], [], []
    A, C = eye, torch.zeros(b, hv, dk, dv, dtype=dt, device=dev)
    for c0 in range(0, t, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, t))
        z = torch.zeros(b, hv, dk, dv, dtype=dt, device=dev)
        _, Bi, cache = reference._gdn_chunk_fwd(qn[:, sl], kn[:, sl], vn[:, sl],
                                                bt[:, sl], gt[:, sl], z)
        ai = torch.exp(cache["glast"]).unsqueeze(-1).unsqueeze(-1) * eye  # WRONG: no -R^T W
        caches.append((sl, cache))
        ais.append(ai)
        pref.append((A, C))
        A, C = ai @ A, ai @ C + Bi
    g = [torch.zeros_like(x) for x in (qn, kn, vn, bt, gt)]
    for i in range(len(caches) - 1, -1, -1):
        sl, cache = caches[i]
        pA, pC = pref[i]
        gai = ga @ pA.mT + gb @ pC.mT
        d = reference._gdn_chunk_bwd(torch.zeros_like(vn[:, sl]), gb, qn[:, sl],
                                     kn[:, sl], vn[:, sl], bt[:, sl], cache, d_ai=gai)
        g[0][:, sl] += d[0]
        g[1][:, sl] += d[1]
        g[2][:, sl] += d[2]
        g[3][:, sl] += d[3]
        g[4][:, sl] += d[4]
        ga, gb = ais[i].mT @ ga, ais[i].mT @ gb
    return g


def _run(decay_a: bool) -> float:
    from tilerl_kernels import reference

    qn, kn, vn, bt, gt = _inputs()
    A, Bb = _span(qn, kn, vn, bt, gt)
    torch.manual_seed(23)
    gA, gB = torch.randn_like(A), torch.randn_like(Bb)
    grads = (_decay_span_bwd(gA, gB, qn, kn, vn, bt, gt) if decay_a
             else reference.gdn_span_ab_bwd(gA, gB, qn, kn, vn, bt, gt, chunk=CHUNK))
    names = ("qn", "kn", "vn", "bt", "gt")

    def loss(name, x):
        d = dict(zip(names, (qn, kn, vn, bt, gt)))
        d[name] = x
        a, bb = _span(d["qn"], d["kn"], d["vn"], d["bt"], d["gt"])
        return float((gA * a).sum() + (gB * bb).sum())

    worst = 0.0
    for name, x0, an in zip(names, (qn, kn, vn, bt, gt), grads):
        nums, ans = [], []
        for j in torch.randperm(an.numel())[:NSAMP]:
            xp, xm = x0.clone(), x0.clone()
            xp.reshape(-1)[j] += STEP
            xm.reshape(-1)[j] -= STEP
            nums.append((loss(name, xp) - loss(name, xm)) / (2 * STEP))
            ans.append(an.reshape(-1)[j].item())
        nums, ans = torch.tensor(nums), torch.tensor(ans)
        if name == "qn":
            assert an.abs().max().item() == 0.0  # q never enters the transfer
            continue
        # qn is structurally absent from the transfer; compare the rest by RMS-rel.
        worst = max(worst, (nums - ans).norm().item() / max(nums.norm().item(), 1e-12))
    return worst


def test_gdn_span_ab_bwd_matches_central_differences():
    assert _run(False) < TOL


def test_decay_a_control_is_red():
    assert _run(True) >= TOL


if __name__ == "__main__":
    decay_a = "--decay-a" in sys.argv
    w = _run(decay_a)
    if decay_a:
        print("decay-a control:", "correctly FAILED" if w >= TOL else "PASSED -- vacuous gate",
              f"{w:.2e}")
        raise SystemExit(0 if w >= TOL else 1)
    print("span transfer reverse matches central differences" if w < TOL
          else f"span reverse FAILED {w:.2e} >= {TOL:.0e}", f"{w:.2e}")
    raise SystemExit(0 if w < TOL else 1)
