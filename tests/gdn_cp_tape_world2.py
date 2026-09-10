"""GDN context-parallel TAPE: world2 end-to-end numerical gradcheck, conv kernel 1.

``gdn_cp_gradcheck_world2`` pins the affine scan reverse in isolation; this pins the
whole CP layer wired through the hand-written tape — the span ``(A, B)``, the exclusive
prefix scan, and the ``s_in = a_pre @ state + b_pre`` start state all recorded as ONE
``gdn_cp`` op and replayed by :func:`gdn_cp_bwd`. Before that op existed the CP forward
ran raw matmul/stack glue the tape could not see, so its state gradient was silently
absent (the failure class the rmsnorm_f32_tape gate documents).

The fixture is conv kernel 1 on purpose: every chunk then starts from a ZERO halo, so
the only cross-rank coupling is the affine transfer under test. The 27B runs kernel 4;
the cross-rank conv-halo adjoint (a window-aware prep reverse plus a cp_halo
reduce-scatter) is the immediately following PR — until it lands q/k/v under kernel 4
are not gradchecked, and ``gdn_cp_bwd`` raises on a nonzero window rather than guess.

The oracle is independent single-process code: a virtual CP that runs the four chunks
from their sequential exclusive prefixes (no torch.distributed) and differentiates the
GLOBAL loss summed over every rank's chunks by central differences. A replicated leaf
(the initial state and the layer params) has one value on both ranks, so its analytic
gradient is the SUM of both ranks' tape grads; a token slice is graded by the rank that
owns it. Perturbing one rank's k/v/gate/beta changes LATER prefixes returned on the
other rank — a rank-local objective would miss exactly the term the scan's cotangent
all-gather carries, so the oracle and the analytic both sum the world.

Controls corrupt only the reverse the tape replays (monkeypatch affine_prefix_scan_bwd
the production gdn_cp_bwd calls; forward stays the real one), and must be caught red:
  --decay-a   scanned A treated as the decay scalar*I in the reverse recurrence
  --no-scan   the span cotangent is dropped (chunks carry no state back to their spans)

    TILERL_TARGET=cpu python3 tests/gdn_cp_tape_world2.py            # the gate
    TILERL_TARGET=cpu python3 tests/gdn_cp_tape_world2.py --decay-a  # red
    TILERL_TARGET=cpu python3 tests/gdn_cp_tape_world2.py --no-scan   # red
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, HV, DK, DV = 1, 8, 2, 6, 6
CHUNK = T // (2 * CP)  # tokens per chunk; rank r holds chunks r and 2cp-1-r
NCHUNK = T // CHUNK
STEP = 1e-3
TOL = 8e-3
NSAMP = 24
PORT = "29541"

LOC_LEAVES = ("q", "k", "v", "g", "beta", "z")
REP_LEAVES = ("state", "conv1d_weight", "dt_bias", "a_log", "norm_weight")


def _chunks_of(rank: int) -> list[int]:
    return [rank, 2 * CP - 1 - rank]


def _make():
    """Deterministic shared inputs, identical on every rank and in the oracle."""
    torch.manual_seed(7)
    qd = HV * DK
    x = {
        "q": torch.randn(B, T, qd) * 0.3,
        "k": torch.randn(B, T, qd) * 0.3,
        "v": torch.randn(B, T, HV * DV) * 0.3,
        "g": torch.randn(B, T, HV),
        "beta": torch.randn(B, T, HV),
        "z": torch.randn(B, T, HV * DV),
        "state": torch.zeros(B, HV, DK, DV),
        "conv1d_weight": torch.randn(3 * qd, 1),
        "dt_bias": torch.randn(HV),
        "a_log": torch.randn(HV),
        "norm_weight": torch.ones(DV),
    }
    torch.manual_seed(9)
    go = torch.randn(B, T, HV * DV)
    return x, go


def _span(reference, x, c):
    sl = slice(c * CHUNK, (c + 1) * CHUNK)
    return reference.gdn_span_ab_raw(
        x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], x["g"][:, sl], x["beta"][:, sl],
        x["state"].shape, conv1d_weight=x["conv1d_weight"], dt_bias=x["dt_bias"],
        a_log=x["a_log"], conv_window=None)


def _global_loss(reference, x, go):
    """Virtual CP: every chunk run from its sequential exclusive prefix, no collective."""
    eye = torch.eye(DK).expand(B, HV, DK, DK).contiguous()
    A, C = eye, torch.zeros(B, HV, DK, DV)
    total = 0.0
    for c in range(NCHUNK):
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        o, _, _ = reference.gdn_forward(
            x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], x["g"][:, sl], x["beta"][:, sl],
            A @ x["state"] + C, z=x["z"][:, sl], conv1d_weight=x["conv1d_weight"],
            dt_bias=x["dt_bias"], a_log=x["a_log"], norm_weight=x["norm_weight"],
            conv_window=None)
        total += float((go[:, sl] * o).sum())
        ai, bi = _span(reference, x, c)
        A, C = ai @ A, ai @ C + bi
    return total


def _finite_diff(reference, x, go, name, chunk, flat_idx):
    xp = {k: (t.clone() if torch.is_tensor(t) else t) for k, t in x.items()}
    xm = {k: (t.clone() if torch.is_tensor(t) else t) for k, t in x.items()}
    if chunk is None:
        xp[name].reshape(-1)[flat_idx] += STEP
        xm[name].reshape(-1)[flat_idx] -= STEP
    else:
        per_token = x[name].shape[-1]
        tok, col = divmod(flat_idx, per_token)
        pos = chunk * CHUNK + tok
        xp[name][:, pos, :].reshape(-1)[col] += STEP
        xm[name][:, pos, :].reshape(-1)[col] -= STEP
    return (_global_loss(reference, xp, go) - _global_loss(reference, xm, go)) / (2 * STEP)


def _install_control(reference, control):
    """Corrupt ONLY the reverse gdn_cp_bwd replays; the forward is untouched."""
    real = reference.affine_prefix_scan_bwd

    def decay_a(goa, gob, a, b, pg, rank, world, chunk_ids=None):
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        d = torch.diagonal(a, dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
        return real(goa, gob, d * eye, b, pg, rank, world, chunk_ids)

    def no_scan(goa, gob, a, b, pg, rank, world, chunk_ids=None):
        return torch.zeros_like(a), torch.zeros_like(b)

    reference.affine_prefix_scan_bwd = decay_a if control == "decay_a" else no_scan


def _rank(rank: int, control: str | None, q) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", PORT)
    os.environ["TILERL_TARGET"] = "cpu"
    try:
        import torch.distributed as dist
        from tilerl_kernels import reference

        from tilerl.autograd import RecordingBackend, Tape
        from tilerl.tensor_parallel import Mesh
        from tilerl.testing import RefBackend

        if control:
            _install_control(reference, control)
        x, go = _make()
        backend = RefBackend()
        mesh = Mesh(cp=CP, rank=rank)
        backend.init_tp(CP, rank, cp_groups=[mesh.cp_group()])
        mine = _chunks_of(rank)
        rec = RecordingBackend(backend)

        loc = {nm: torch.cat([x[nm][:, c * CHUNK:(c + 1) * CHUNK] for c in mine], dim=1)
               for nm in LOC_LEAVES}
        go_loc = torch.cat([go[:, c * CHUNK:(c + 1) * CHUNK] for c in mine], dim=1)

        tape = Tape()
        with tape:
            rec.gdn_cp(
                loc["q"], loc["k"], loc["v"], loc["g"], loc["beta"], x["state"],
                z=loc["z"], conv1d_weight=x["conv1d_weight"], dt_bias=x["dt_bias"],
                a_log=x["a_log"], norm_weight=x["norm_weight"],
                conv_windows=[None, None], chunk_ids=mine)
        grads = tape.backward(go_loc)

        row = {"mine": mine}
        for nm in LOC_LEAVES:
            gv = grads.get(id(loc[nm]))
            row[nm] = {str(mine[i]): None if gv is None else
                       gv[:, i * CHUNK:(i + 1) * CHUNK].reshape(-1).tolist()
                       for i in range(2)}
        for nm in REP_LEAVES:
            gv = grads.get(id(x[nm]))
            row[nm] = None if gv is None else gv.reshape(-1).tolist()
        dist.barrier()
        q.put((rank, row))
    except BaseException:
        import traceback

        q.put((rank, {"error": traceback.format_exc()}))


def _entry(rank, control, q):
    _rank(rank, control, q)


def main() -> int:
    control = "decay_a" if "--decay-a" in sys.argv else ("no_scan" if "--no-scan" in sys.argv else None)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_entry, args=(r, control, q)) for r in range(CP)]
    for p in ps:
        p.start()
    rows = {}
    for _ in range(CP):
        rank, row = q.get(timeout=180)
        rows[rank] = row
    for p in ps:
        p.join(30)

    for r in range(CP):
        if "error" in rows.get(r, {}):
            print(rows[r]["error"])
            return 1

    from tilerl_kernels import reference

    x, go = _make()
    # RMS-rel over the sampled vector: a grad with many near-zero entries makes a
    # per-element max-rel meaningless (FD noise on a ~0 component), the same reason the
    # span gate compares by norm rather than the worst element.
    worst = 0.0

    def rms_rel(nums, ans):
        nums = torch.tensor(nums)
        ans = torch.tensor(ans)
        return (nums - ans).norm().item() / max(nums.norm().item(), 1e-9)

    for nm in LOC_LEAVES:
        per_tok = x[nm].shape[-1]
        nums, ans = [], []
        for c in range(NCHUNK):
            owner = 0 if c in _chunks_of(0) else 1
            an_list = rows[owner][nm][str(c)]
            for j in torch.randperm(CHUNK * per_tok)[:NSAMP]:
                j = int(j)
                nums.append(_finite_diff(reference, x, go, nm, c, j))
                ans.append(float(an_list[j]))
        w = rms_rel(nums, ans)
        worst = max(worst, w)
        print(f"  loc {nm:6s} {w:.2e}")

    for nm in REP_LEAVES:
        n = len(rows[0][nm])
        nums, ans = [], []
        for j in torch.randperm(n)[:NSAMP]:
            j = int(j)
            ans.append(float(rows[0][nm][j]) + float(rows[1][nm][j]))
            nums.append(_finite_diff(reference, x, go, nm, None, j))
        w = rms_rel(nums, ans)
        worst = max(worst, w)
        print(f"  rep {nm:14s} {w:.2e}")

    if control:
        red = worst >= TOL
        print(f"{control} control rel {worst:.2e}:",
              "correctly FAILED" if red else "PASSED -- vacuous gate")
        return 0 if red else 1
    ok = worst < TOL
    print("gdn_cp world2 tape matches global central differences" if ok
          else f"gdn_cp world2 tape FAILED {worst:.2e} >= {TOL:.0e}", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
