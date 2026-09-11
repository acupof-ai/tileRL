"""Sparse paged_attention kwarg port: the tilelang Backend's concat-table remap
must equal the RefBackend CPU twin token-for-token.

Cases cover the mask geometry from the #507 contract:
- prefill chunk whose own span OVERLAPS the previous chunk (prefill_from off a 16
  boundary): selected pages unmasked, own causal by global position;
- decode trailing 8-page own window with selected earlier pages and S=1;
- ragged per-row n_sel and a SELECTED physical block id 0 (the right-pad trap:
  n_sel is the sole validity marker).

  TILERL_TARGET=cpu python3 scripts/probe_sparse_attn_kwargs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages/tilerl-kernels/src"))

import torch  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.testing import RefBackend  # noqa: E402

BLK = 16
HQ, HKV, D = 4, 2, 32
NB = 16
SCALE = 1.0 / (D ** 0.5)


def run_case(name, rows, k_pool, v_pool):
    """rows: list of dicts with s, sel(phys ids), own(phys ids), own_len,
    own_offsets, q_start. Batches B=len(rows); s must match across rows."""
    b = len(rows)
    s = rows[0]["s"]
    assert all(r["s"] == s for r in rows)
    dev = k_pool.device
    omax = max(len(r["own"]) for r in rows)
    pmax = max([len(r["sel"]) for r in rows] + [1])
    q = torch.randn(b, s, HQ, D, device=dev) * 0.3
    own_table = torch.zeros(b, omax, dtype=torch.long, device=dev)
    page_sel = torch.zeros(b, pmax, dtype=torch.long, device=dev)
    n_sel = torch.zeros(b, dtype=torch.long, device=dev)
    own_lens = torch.zeros(b, dtype=torch.long, device=dev)
    own_offsets = torch.zeros(b, dtype=torch.long, device=dev)
    q_start = torch.zeros(b, dtype=torch.long, device=dev)
    seq_lens = torch.zeros(b, dtype=torch.long, device=dev)
    for i, r in enumerate(rows):
        own_table[i, : len(r["own"])] = torch.tensor(r["own"], device=dev)
        page_sel[i, : len(r["sel"])] = torch.tensor(r["sel"], device=dev)
        n_sel[i] = len(r["sel"])
        own_lens[i] = r["own_len"]
        own_offsets[i] = r["own_offsets"]
        q_start[i] = r["q_start"]
        seq_lens[i] = r["q_start"] + s  # twin only reads this when q_start absent
    kw = dict(page_sel=page_sel, n_sel=n_sel, own_lens=own_lens,
              own_offsets=own_offsets, q_start=q_start)
    ref = RefBackend().paged_attention(
        q, k_pool, v_pool, own_table, seq_lens, SCALE, seq_q_lens=None, **kw)
    be = get_backend()
    # dense None kwargs must stay the dense path (negative control on the gate)
    got = be.paged_attention(
        q, k_pool, v_pool, own_table, seq_lens, SCALE, seq_q_lens=None, **kw)
    rel = (got.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
    print(f"{name:24s} max rel {rel:.3e}")
    assert rel < 2e-2, (name, rel)
    return rel


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k_pool = torch.randn(NB, HKV, BLK, D, device=dev) * 0.2
    v_pool = torch.randn(NB, HKV, BLK, D, device=dev) * 0.2

    # Case A: prefill chunk, own span overlaps the previous chunk.
    #   prefill_from=q_start=35 (page 2 starts at 32 -> 3 overlap tokens), S=20,
    #   q_hi=55 -> own pages 2,3 own_len=23. Earlier pages 0,1 selected but mapped
    #   to physical [7,0] (physical 0 deliberately selected). Own phys [5,9].
    prefill = dict(
        s=20, sel=[7, 0], own=[5, 9], own_len=23, own_offsets=32, q_start=35)
    # ragged second row: only ONE selected page (id 0), two OWN blocks (own_len=20
    # -> ceil(20/16)=2 own physical ids): own_offsets=48 q_start=48 S=20.
    prefill_b = dict(
        s=20, sel=[0], own=[11, 15], own_len=20, own_offsets=48, q_start=48)
    run_case("prefill overlap/ragged", [prefill, prefill_b], k_pool, v_pool)

    # Case B: decode trailing window, S=1, seq_len=200.
    #   own pages 5..12 (8), own_offsets=80, own_len=120, q_start=199; three
    #   earlier pages selected [13, 0, 2].
    decode = dict(
        s=1, sel=[13, 0, 2], own=[1, 3, 4, 6, 8, 10, 12, 14],
        own_len=120, own_offsets=80, q_start=199)
    # second decode row: no selections (short context), own_len=40 from page 0 ->
    # 3 own blocks (32 + 8 tokens), q_start=39.
    decode_b = dict(
        s=1, sel=[], own=[5, 7, 2], own_len=40, own_offsets=0, q_start=39)
    run_case("decode window/ragged", [decode, decode_b], k_pool, v_pool)

    print("PROBE_OK")


if __name__ == "__main__":
    main()
