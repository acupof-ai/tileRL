"""GDN context-parallel conv HALO: world2 tape gradcheck at conv kernel 4.

``gdn_cp_tape_world2.py`` covers the affine state transfer with conv kernel 1 (zero
halo). This covers the other cross-rank coupling the 27B actually has: with conv kernel
4 each chunk's prep reads the predecessor chunk's last K-1=3 RAW qkv rows, exchanged by
``cp_halo``. The windows here come from a REAL ``backend.cp_halo`` all-gather, not a
synthetic list — the reverse under test routes their cotangent back onto the
predecessor's chunk tail with :func:`cp_halo_bwd`, which must also handle rank 1's LOCAL
predecessor (zigzag cp=2 gives rank 1 chunks 1 and 2, so chunk 2's halo is its own chunk
1 tail), not just the remote one.

Oracle: a single-process virtual CP that prepends each chunk's true predecessor tail and
runs the chunk (and its span) with that window, scanning states as in the kernel-1 gate.
Analytic runs the real Tape + RecordingBackend.gdn_cp. Floats over a spawn Queue.

Controls corrupt only the reverse:
  --no-halo  cp_halo_bwd drops every window grad (q/k/v tails then miss their cross-chunk
             and cross-rank terms) — must be caught red.

    TILERL_TARGET=cpu python3 tests/gdn_cp_halo_tape_world2.py             # the gate
    TILERL_TARGET=cpu python3 tests/gdn_cp_halo_tape_world2.py --no-halo   # red
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

CP = 2
B, T, HV, DK, DV = 1, 16, 2, 6, 6
KERNEL = 4
WIDTH = KERNEL - 1
CHUNK = 4             # tokens per chunk (>= WIDTH); 4 chunks over 2 ranks
NCHUNK = T // CHUNK
STEP = 1e-3
TOL = 8e-3
NSAMP = 10
PORT = "29543"

LOC_LEAVES = ("q", "k", "v", "g", "beta", "z")
REP_LEAVES = ("state", "conv1d_weight", "dt_bias", "a_log", "norm_weight")


def _chunks_of(rank: int) -> list[int]:
    return [rank, 2 * CP - 1 - rank]


def _window_of(x, c):
    """The conv window chunk c really receives: its predecessor chunk's last WIDTH raw
    qkv rows (None for chunk 0). Single-process oracle truth."""
    if c == 0:
        return None
    qkv = torch.cat([x["q"], x["k"], x["v"]], dim=-1)
    return qkv[:, c * CHUNK - WIDTH : c * CHUNK]


def _make():
    torch.manual_seed(13)
    qd = HV * DK
    x = {
        "q": torch.randn(B, T, qd) * 0.3,
        "k": torch.randn(B, T, qd) * 0.3,
        "v": torch.randn(B, T, HV * DV) * 0.3,
        "g": torch.randn(B, T, HV),
        "beta": torch.randn(B, T, HV),
        "z": torch.randn(B, T, HV * DV),
        "state": torch.zeros(B, HV, DK, DV),
        "conv1d_weight": torch.randn(3 * qd, KERNEL) * 0.3,
        "dt_bias": torch.randn(HV),
        "a_log": torch.randn(HV),
        "norm_weight": torch.ones(DV),
    }
    torch.manual_seed(17)
    go = torch.randn(B, T, HV * DV)
    return x, go


def _span(reference, x, c):
    sl = slice(c * CHUNK, (c + 1) * CHUNK)
    return reference.gdn_span_ab_raw(
        x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], x["g"][:, sl], x["beta"][:, sl],
        x["state"].shape, conv1d_weight=x["conv1d_weight"], dt_bias=x["dt_bias"],
        a_log=x["a_log"], conv_window=_window_of(x, c))


def _global_loss(reference, x, go):
    eye = torch.eye(DK).expand(B, HV, DK, DK).contiguous()
    A, C = eye, torch.zeros(B, HV, DK, DV)
    total = 0.0
    for c in range(NCHUNK):
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        win = _window_of(x, c)
        o, _, _ = reference.gdn_forward(
            x["q"][:, sl], x["k"][:, sl], x["v"][:, sl], x["g"][:, sl], x["beta"][:, sl],
            A @ x["state"] + C, z=x["z"][:, sl], conv1d_weight=x["conv1d_weight"],
            dt_bias=x["dt_bias"], a_log=x["a_log"], norm_weight=x["norm_weight"],
            conv_window=win, chunkwise=CHUNK)
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


def _rank(rank: int, no_halo: bool, q) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", PORT)
    os.environ["TILERL_TARGET"] = "cpu"
    try:
        import torch.distributed as dist
        from tilerl_kernels import reference

        from tilerl.autograd import RecordingBackend, Tape
        from tilerl.tensor_parallel import Mesh
        from tilerl.testing import RefBackend

        if no_halo:  # control: feed ZERO window grads into the (still symmetric) gather,
            # so q/k/v tails miss every cross-chunk and cross-rank halo term. The collective
            # still runs on both ranks with the same shape — a dropped all_gather deadlocks.
            real_halo_bwd = reference.cp_halo_bwd

            def _zero_halo(g_windows, qkv, pg, rank, world, chunk_ids, width):
                return real_halo_bwd([None] * len(g_windows), qkv, pg, rank, world,
                                     chunk_ids, width)

            reference.cp_halo_bwd = _zero_halo

        x, go = _make()
        backend = RefBackend()
        mesh = Mesh(cp=CP, rank=rank)
        backend.init_tp(CP, rank, cp_groups=[mesh.cp_group()])
        mine = _chunks_of(rank)
        ids_by_rank = [_chunks_of(r) for r in range(CP)]
        rec = RecordingBackend(backend)

        loc = {nm: torch.cat([x[nm][:, c * CHUNK:(c + 1) * CHUNK] for c in mine], dim=1)
               for nm in LOC_LEAVES}
        go_loc = torch.cat([go[:, c * CHUNK:(c + 1) * CHUNK] for c in mine], dim=1)

        # REAL cp_halo exchange, exactly as model._gdn_cp builds it.
        stack = torch.stack([torch.cat([loc["q"][:, i * CHUNK:(i + 1) * CHUNK],
                                        loc["k"][:, i * CHUNK:(i + 1) * CHUNK],
                                        loc["v"][:, i * CHUNK:(i + 1) * CHUNK]], dim=-1)
                             for i in range(len(mine))])
        halos = backend.cp_halo(stack, ids_by_rank, WIDTH)

        tape = Tape()
        with tape:
            rec.gdn_cp(
                loc["q"], loc["k"], loc["v"], loc["g"], loc["beta"], x["state"],
                z=loc["z"], conv1d_weight=x["conv1d_weight"], dt_bias=x["dt_bias"],
                a_log=x["a_log"], norm_weight=x["norm_weight"],
                conv_windows=halos, chunk_ids=mine, chunk=CHUNK)
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


def _entry(rank, no_halo, q):
    _rank(rank, no_halo, q)


def main() -> int:
    no_halo = "--no-halo" in sys.argv
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_entry, args=(r, no_halo, q)) for r in range(CP)]
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

    def rms_rel(nums, ans):
        nums = torch.tensor(nums)
        ans = torch.tensor(ans)
        return (nums - ans).norm().item() / max(nums.norm().item(), 1e-9)

    worst = 0.0
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

    if no_halo:
        red = worst >= TOL
        print(f"no-halo control rel {worst:.2e}:",
              "correctly FAILED" if red else "PASSED -- vacuous gate")
        return 0 if red else 1
    ok = worst < TOL
    print("gdn_cp world2 conv-halo tape matches global central differences" if ok
          else f"gdn_cp halo tape FAILED {worst:.2e} >= {TOL:.0e}", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
