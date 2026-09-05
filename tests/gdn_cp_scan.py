"""GDN under CP: the chunk scan composes, so a rank can start from a prefix.

The recurrence in ``_gdn_chunk_fwd`` is affine in the incoming state::

    d      = U - W s
    s_next = exp(glast) s + R^T d = (exp(glast) I - R^T W) s + R^T U
             \\_______________  _______________/   \\___  ___/
                             A                          B

so a rank holding a later chunk does not need its predecessor's state, only the
composed (A, B) of every chunk before it -- and (A2, B2) o (A1, B1) is
(A2 A1, A2 B1 + B2), which is associative, so the prefixes come from a scan.

The trap this gate exists for: **A is a DK x DK operator, not the decay
scalar.** ``d`` depends on ``s``, and folding that dependence in is what makes
the recurrence affine. Reading it as "decay the state, add the chunk's
contribution" is wrong by 23% and produces a plausible loss.

    TILERL_TARGET=cpu python3 tests/gdn_cp_scan.py            # the gate
    TILERL_TARGET=cpu python3 tests/gdn_cp_scan.py --decay-a  # control: A as a scalar
    TILERL_TARGET=cpu python3 tests/gdn_cp_scan.py --no-compose  # control: skip the prefix
"""

from __future__ import annotations

import sys

import torch

sys.path[:0] = ["src", "packages/tilerl-kernels/src"]

from tilerl_kernels.reference import _gdn_chunk_fwd  # noqa: E402

B, HV, DK, DV, CHUNK, NCHUNK = 1, 4, 8, 8, 8, 4
T = CHUNK * NCHUNK


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    return (torch.randn(B, T, HV, DK), torch.randn(B, T, HV, DK), torch.randn(B, T, HV, DV),
            -torch.rand(B, T, HV) * 0.1, torch.rand(B, T, HV))


def chunk_ab(qc, kc, vc, bc, gtc, decay_a: bool):
    """(A, B) for one chunk. B is the chunk run from a zero state; A is the
    operator acting on whatever state actually arrives."""
    zero = torch.zeros(B, HV, DK, DV)
    _, b_term, c = _gdn_chunk_fwd(qc, kc, vc, bc, gtc, zero)
    eye = torch.eye(DK).expand(B, HV, DK, DK)
    decay = torch.exp(c["glast"]).unsqueeze(-1).unsqueeze(-1) * eye
    if decay_a:  # the control: d treated as independent of s
        return decay, b_term
    return decay - c["R"].transpose(-1, -2) @ c["W"], b_term


def compose(x, y):
    """(A1,B1) then (A2,B2) -> the single affine map equal to both in sequence."""
    (a1, b1), (a2, b2) = x, y
    return a2 @ a1, a2 @ b1 + b2


def main() -> int:
    decay_a = "--decay-a" in sys.argv
    no_compose = "--no-compose" in sys.argv
    qn, kn, v, gt, bt = _inputs()
    s0 = torch.randn(B, HV, DK, DV) * 0.1

    # Truth: the chunks in sequence, each starting where the last ended.
    truth, s = [], s0
    for i in range(NCHUNK):
        sl = slice(i * CHUNK, (i + 1) * CHUNK)
        out, s, _ = _gdn_chunk_fwd(qn[:, sl], kn[:, sl], v[:, sl], bt[:, sl], gt[:, sl], s)
        truth.append((out, s))

    # CP: every rank builds its own (A, B) with no state, an exclusive scan gives
    # each the prefix before it, and each re-runs its chunk from that prefix.
    abs_ = [chunk_ab(qn[:, slice(i * CHUNK, (i + 1) * CHUNK)],
                     kn[:, slice(i * CHUNK, (i + 1) * CHUNK)],
                     v[:, slice(i * CHUNK, (i + 1) * CHUNK)],
                     bt[:, slice(i * CHUNK, (i + 1) * CHUNK)],
                     gt[:, slice(i * CHUNK, (i + 1) * CHUNK)], decay_a)
            for i in range(NCHUNK)]

    eye = torch.eye(DK).expand(B, HV, DK, DK)
    prefix, acc = [], (eye, torch.zeros(B, HV, DK, DV))
    for i in range(NCHUNK):
        prefix.append(acc)  # exclusive: chunk i sees everything BEFORE it
        acc = compose(acc, abs_[i])

    ok, worst_out, worst_s, per_chunk = True, 0.0, 0.0, []
    for i in range(NCHUNK):
        a, b = prefix[i]
        s_in = s0 if no_compose else a @ s0 + b  # the control: ignore the prefix
        sl = slice(i * CHUNK, (i + 1) * CHUNK)
        out, s_out, _ = _gdn_chunk_fwd(qn[:, sl], kn[:, sl], v[:, sl], bt[:, sl], gt[:, sl], s_in)
        t_out, t_s = truth[i]
        # Relative: |s| runs to ~1e3 in f32, so an absolute bound tight at |s|=1
        # flags plain rounding at |s|=1000 (the same ulp mistake as tp_world2).
        r_out = (out - t_out).abs().max().item() / max(t_out.abs().max().item(), 1e-9)
        r_s = (s_out - t_s).abs().max().item() / max(t_s.abs().max().item(), 1e-9)
        worst_out, worst_s = max(worst_out, r_out), max(worst_s, r_s)
        per_chunk.append(r_s)
        if max(r_out, r_s) > 1e-6:
            ok = False

    # Per chunk, because both controls saturate near 1.0 on the worst chunk and a
    # saturated number cannot tell them apart. Chunk 0's prefix is the identity,
    # so a correct chunk 0 with a wrong chunk 1 means the COMPOSE is wrong, while
    # a wrong chunk 0 would mean the chunk math itself is.
    print("cp scan over " + str(NCHUNK) + " chunks: state rel per chunk "
          + " ".join(f"{r:.1e}" for r in per_chunk))
    print(f"  worst: out rel {worst_out:.2e}, state rel {worst_s:.2e}")
    if decay_a or no_compose:
        which = "decay-a" if decay_a else "no-compose"
        print(f"{which} control:", "correctly FAILED" if not ok else "PASSED -- vacuous gate")
        return 0 if not ok else 1
    print("gdn: a chunk started from its composed prefix matches the sequential scan"
          if ok else "gdn: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
