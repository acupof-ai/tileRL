"""GDN context-parallel prefix scan: NUMERIC GRADCHECK for the affine reverse.

The forward gate ``gdn_world2`` proves a chunk started from its scanned prefix
matches the sequential scan. It cannot see the backward: ``cp_prefix_scan`` maps
to ``reference.affine_prefix_scan``, which had no reverse, so under CP the tape
silently dropped the scan's gradient (the same failure class the
``rmsnorm_f32`` gate documents — an op absent from ``autograd._BWD`` records
nothing). This gate pins the reverse before CP training trusts it:

    dA_k = Q P_k^T + R C_k^T
    dB_k = R
    Q <- A_k^T Q + X_k ; R <- A_k^T R + Y_k

over a reverse pass, where (P_k, C_k) is the forward's exclusive prefix and
(X_k, Y_k) the gathered cotangent of the output pair at k.

The oracle is independent code: the GLOBAL loss summed over ALL ranks' outputs
under a plain sequential exclusive scan, differentiated by central differences.
A pair on this rank shapes prefixes returned on the OTHER rank, so the oracle
perturbs one local pair and measures the summed loss — differentiating a
rank-local-only objective would miss the cross-rank term the cotangent gather
exists to carry.

Controls are the same two conceptual errors as the forward gate, each run as a
"backward" that must be caught red:
  --decay-a   reverse treats A as the decay scalar (drops the R^T W operator)
  --no-compose reverse skips the prefix (P=I, C=0) and the cotangent gather

    TILERL_TARGET=cpu python3 tests/gdn_cp_gradcheck_world2.py           # the gate
    TILERL_TARGET=cpu python3 tests/gdn_cp_gradcheck_world2.py --decay-a  # red
    TILERL_TARGET=cpu python3 tests/gdn_cp_gradcheck_world2.py --no-compose# red
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, HV, DK, DV, CHUNK = 1, 8, 2, 6, 4, 2
NCHUNK = T // CHUNK  # 4 chunks over 2 ranks: rank r holds r and NCHUNK-1-r
STEP = 1e-2            # f32 floor dominates below this (measured)
TOL = 5e-2
NSAMP = 12


def _chunks_of(rank: int) -> list[int]:
    return [rank, NCHUNK - 1 - rank]


def _global_pairs():
    """The (A, B) of every chunk in sequence order, identical on every rank."""
    from tilerl_kernels import reference

    torch.manual_seed(11)
    q = torch.randn(B, T, HV, DK)
    k = torch.randn(B, T, HV, DK)
    v = torch.randn(B, T, HV, DV)
    bt = -torch.rand(B, T, HV) * 0.1
    gt = torch.rand(B, T, HV)
    A, Bb = [], []
    for c in range(NCHUNK):
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        a, b = reference.gdn_span_ab(q[:, sl], k[:, sl], v[:, sl], bt[:, sl], gt[:, sl],
                                     chunk=CHUNK)
        A.append(a)
        Bb.append(b)
    return torch.stack(A), torch.stack(Bb)


def _global_cotangents():
    torch.manual_seed(23)
    return (torch.randn(NCHUNK, B, HV, DK, DK),
            torch.randn(NCHUNK, B, HV, DK, DV))


def _global_loss(A, Bb, GT, HT):
    """Sum over ranks of (G_c . P_c + H_c . C_c) under a sequential exclusive scan."""
    eye = torch.eye(DK, dtype=A.dtype).expand(B, HV, DK, DK).contiguous()
    pa, pc, total = eye, torch.zeros_like(Bb[0]), 0.0
    for c in range(NCHUNK):
        total = total + (GT[c] * pa).sum() + (HT[c] * pc).sum()
        pa, pc = A[c] @ pa, A[c] @ pc + Bb[c]
    return total


def _decay_pairs(A):
    """The forward control's wrong model: A as the per-chunk decay scalar*I."""
    out = torch.zeros_like(A)
    for c in range(NCHUNK):
        d = torch.diagonal(A[c], dim1=-2, dim2=-1).mean(dim=-1)  # B,HV mean of diagonal
        out[c] = d.view(B, HV, 1, 1) * torch.eye(DK).expand(B, HV, DK, DK)
    return out


def _rank(rank: int, decay_a: bool, no_compose: bool) -> dict:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    os.environ["TILERL_TARGET"] = "cpu"

    import torch.distributed as dist
    from tilerl_kernels import reference

    dist.init_process_group("gloo", rank=rank, world_size=CP)
    pg = dist.group.WORLD
    A, Bb = _global_pairs()
    GT, HT = _global_cotangents()
    mine = _chunks_of(rank)

    loc_a = torch.stack([A[c] for c in mine]).contiguous()
    loc_b = torch.stack([Bb[c] for c in mine]).contiguous()
    goa = torch.stack([GT[c] for c in mine]).contiguous()
    gob = torch.stack([HT[c] for c in mine]).contiguous()

    # Analytic gradient under test. The controls corrupt ONLY the reverse's model
    # of the operator; the forward/pairs and the oracle stay the real ones, so a
    # match would mean the gate cannot see the error.
    if decay_a:
        ga, gb = _wrong_scalar_reverse(reference, loc_a, loc_b, goa, gob, pg, rank, mine)
    elif no_compose:
        ga, gb = _wrong_no_compose_reverse(reference, loc_a, loc_b, goa, gob, pg, rank, mine)
    else:
        ga, gb = reference.affine_prefix_scan_bwd(goa, gob, loc_a, loc_b, pg, rank, CP, mine)

    # Independent central differences on the GLOBAL objective. Perturbing a local
    # pair c moves the prefix returned to the other rank too, so the loss sums all
    # ranks' chunks — a rank-local objective would miss exactly the term the
    # cotangent gather exists to carry.
    rel = {"a": 0.0, "b": 0.0}
    for i, c in enumerate(mine):
        for kind, an_t, base in (("a", ga[i], A), ("b", gb[i], Bb)):
            for j in torch.randperm(an_t.numel())[:NSAMP]:
                fp_t = base.clone()
                fm_t = base.clone()
                fp_t[c].reshape(-1)[j] = base[c].reshape(-1)[j].item() + STEP
                fm_t[c].reshape(-1)[j] = base[c].reshape(-1)[j].item() - STEP
                args_a = fp_t if kind == "a" else A
                args_b = fp_t if kind == "b" else Bb
                f_plus = _global_loss(args_a, args_b, GT, HT)
                args_a = fm_t if kind == "a" else A
                args_b = fm_t if kind == "b" else Bb
                f_minus = _global_loss(args_a, args_b, GT, HT)
                num = (f_plus - f_minus) / (2 * STEP)
                an = an_t.reshape(-1)[j].item()
                rel[kind] = max(rel[kind], abs(num - an) / max(abs(num), abs(an), 1e-9))
    dist.barrier()
    return rel


def _entry(rank: int, decay_a: bool, no_compose: bool, q) -> None:
    """Top-level (picklable) rank entry: forward the result or a traceback."""
    import traceback

    try:
        q.put((rank, _rank(rank, decay_a, no_compose)))
    except BaseException:
        q.put((rank, {"error": traceback.format_exc()}))


def _wrong_scalar_reverse(reference, loc_a, loc_b, goa, gob, pg, rank, mine):
    """dA/dB as if A were the decay scalar I — drops the operator dependence."""
    ca, cb, seq_ids, want, _ = reference._scan_gather_pairs(loc_a, loc_b, pg, rank, CP, mine)
    gpa, gpb, _, _, _ = reference._scan_gather_pairs(goa, gob, pg, rank, CP, mine)
    sa = _decay_pairs(ca)  # the wrong model used ONLY inside the reverse
    eye = torch.eye(DK).expand(B, HV, DK, DK).contiguous()
    pa, pc = eye, torch.zeros_like(cb[0])
    pref = [(pa, pc)]
    for k in range(len(ca)):
        pa, pc = ca[k] @ pa, ca[k] @ pc + cb[k]
        pref.append((pa, pc))
    qa, rb = torch.zeros_like(ca[0]), torch.zeros_like(cb[0])
    ga_all, gb_all = torch.zeros_like(ca), torch.zeros_like(cb)
    for k in range(len(ca) - 1, -1, -1):
        pk, ck = pref[k]
        ga_all[k] = qa @ pk.mT + rb @ ck.mT
        gb_all[k] = rb
        qa = sa[k].mT @ qa + gpa[k]
        rb = sa[k].mT @ rb + gpb[k]
    ga = torch.empty_like(loc_a)
    gb = torch.empty_like(loc_b)
    for k, cid in enumerate(seq_ids):
        if cid in want:
            ga[want[cid]], gb[want[cid]] = ga_all[k], gb_all[k]
    return ga, gb


def _wrong_no_compose_reverse(reference, loc_a, loc_b, goa, gob, pg, rank, mine):
    """No prefix (P=I, C=0) and rank-local cotangents only — the two omissions
    that 'start every chunk from zero' implies on the reverse."""
    ca, cb, seq_ids, want, _ = reference._scan_gather_pairs(loc_a, loc_b, pg, rank, CP, mine)
    # local-only cotangents placed by id, everything else zero
    gpa = torch.zeros_like(ca)
    gpb = torch.zeros_like(cb)
    for i, cid in enumerate(mine):
        k = seq_ids.index(cid)
        gpa[k] = goa[i]
        gpb[k] = gob[i]
    eye = torch.eye(DK).expand(B, HV, DK, DK).contiguous()
    qa, rb = torch.zeros_like(ca[0]), torch.zeros_like(cb[0])
    ga_all, gb_all = torch.zeros_like(ca), torch.zeros_like(cb)
    for k in range(len(ca) - 1, -1, -1):
        ga_all[k] = qa @ eye.mT            # P_k wrongly taken as I
        gb_all[k] = rb                     # C_k wrongly taken as 0
        qa = ca[k].mT @ qa + gpa[k]
        rb = ca[k].mT @ rb + gpb[k]
    ga = torch.empty_like(loc_a)
    gb = torch.empty_like(loc_b)
    for k, cid in enumerate(seq_ids):
        if cid in want:
            ga[want[cid]], gb[want[cid]] = ga_all[k], gb_all[k]
    return ga, gb


def main() -> int:
    decay_a = "--decay-a" in sys.argv
    no_compose = "--no-compose" in sys.argv

    # mp.spawn + mp.Manager hangs the parent on this box at child teardown (gloo +
    # Manager shutdown), so use the same spawn context explicitly with a plain
    # Queue; the workers return their row over it and exit cleanly.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_entry, args=(r, decay_a, no_compose, q)) for r in range(CP)]
    for p in ps:
        p.start()
    out: dict = {}
    for _ in range(CP):
        rank, rel = q.get(timeout=120)
        out[rank] = rel
    for p in ps:
        p.join(30)

    for r in range(CP):
        if "error" in out.get(r, {}):
            print(out[r]["error"])
            return 1

    worst = max(max(v["a"], v["b"]) for v in out.values())
    print(" ".join(f"[r{r}] dA {out[r]['a']:.2e} dB {out[r]['b']:.2e}" for r in range(CP)))
    if decay_a or no_compose:
        which = "decay-a" if decay_a else "no-compose"
        red = worst >= TOL
        print(f"{which} control:", "correctly FAILED" if red else "PASSED -- vacuous gate")
        return 0 if red else 1
    ok = worst < TOL
    print("affine scan reverse matches central differences" if ok
          else f"affine scan reverse FAILED: {worst:.2e} >= {TOL:.0e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
