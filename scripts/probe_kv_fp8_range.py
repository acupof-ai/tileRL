"""What is the per-block dynamic range of real KV data, and what does fp8 cost at that grid?

`docs/design-fp8-kv.md` picks one f32 scale per `(plane, block, head)` -- 16 tokens x
head_dim under one absmax -- and names the one thing it cannot settle from the tree:
e4m3 or e5m2. "The KV plane's dynamic range per 16-token block is not measured anywhere
in this tree", so this measures it: one prefill through the engine, then the pool itself.

Two things it reports, and one it does not:

* the DISTRIBUTION of `absmax / absmin_nonzero` per group, K and V apart. A mean hides
  the case the coarse grid invites -- one spiky block among thousands.
* the round-trip error of `quant_kv_fp8`/`dequant_kv_fp8` on that same pool data, for
  both dtypes, measured rather than derived from mantissa bits. `zeroed` is the concrete
  damage: elements a group's own absmax pushes below the dtype's subnormals.
* NOT a logit or next-token number. That needs the kernel and the bf16 oracle
  (design note section 3); this bounds the pool error the kernel would then carry.

Token ids are random, so the ranges come from the model's weights through the real
prefill path rather than from natural text. Text is the follow-up if the distribution
turns out to depend on it.

    TILERL_TARGET=cpu python3 scripts/probe_kv_fp8_range.py --model tiny --prompt-tokens 64
    CUDA_VISIBLE_DEVICES=6 TILERL_TARGET=cuda python3 scripts/probe_kv_fp8_range.py \
        --model qwen38-27b --source /work/Qwen3.8-27B-NVFP4 --out runs/kv_fp8_range.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tilerl_kernels.reference import dequant_kv_fp8, quant_kv_fp8

from tilerl import config as C
from tilerl.engine import SamplingParams, build_engine
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore, PagedKvPool
from tilerl.model import build_random, load_hf

DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _pct(v: torch.Tensor, p: float) -> float:
    """Nearest-rank percentile. kthvalue, not quantile: quantile refuses >2**24 inputs."""
    k = min(v.numel(), max(1, round(p * v.numel())))
    return v.kthvalue(k).values.item()


def _dist(v: torch.Tensor) -> dict:
    return {"n": int(v.numel()), "min": v.min().item(), "p50": _pct(v, 0.50),
            "p90": _pct(v, 0.90), "p99": _pct(v, 0.99), "max": v.max().item()}


def _ranges(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(plane, block, head) absmax and absmax/absmin_nonzero, groups with data only."""
    a = x.float().abs()
    mx = a.amax((-2, -1))
    mn = torch.where(a > 0, a, torch.inf).amin((-2, -1))
    live = mx > 0
    return mx[live], (mx[live] / mn[live])


def _q_block_head(x: torch.Tensor, dtype: torch.dtype):
    """The grid NOT taken: one scale per (plane, block, head) -- amax over tokens AND head_dim.

    Local to this probe, because the shipped path is per-token; this arm exists only to record
    what the coarse grid would have cost on real KV.
    """
    xf = x.float()
    scale = xf.abs().amax((-2, -1)).clamp_min(1e-12) / torch.finfo(dtype).max
    return (xf / scale[..., None, None]).to(dtype), scale


def _roundtrip(x: torch.Tensor, dtype: torch.dtype, per_token: bool = True) -> dict:
    """What quantizing this pool data to `dtype` costs, at the per-token or block-head grid."""
    xf = x.float()
    if per_token:
        d = dequant_kv_fp8(*quant_kv_fp8(x, dtype))
    else:
        q, sc = _q_block_head(x, dtype)
        d = q.float() * sc[..., None, None]
    e = (d - xf).abs()
    gmax = xf.abs().amax((-2, -1)).clamp_min(1e-30)
    nz = xf != 0
    rel = e[nz] / xf.abs()[nz]
    return {"dtype": str(dtype), "grid": "per_token" if per_token else "block_head",
            "finfo_max": torch.finfo(dtype).max,
            "rel_group_absmax_max": (e.amax((-2, -1)) / gmax).max().item(),
            "rel_elem_p50": _pct(rel, 0.50), "rel_elem_p99": _pct(rel, 0.99),
            "rel_elem_max": rel.max().item(),
            "zeroed_frac": int(((d == 0) & nz).sum()) / int(nz.sum())}


def _build(name: str, source: str, seed: int):
    if name.startswith("tiny"):
        cfg = C.tiny() if name == "tiny" else C.tiny(65536)
        return cfg, build_random(cfg, seed=seed)
    cfg = C.qwen36_27b() if name == "qwen36-27b" else C.qwen38_27b()
    if not source:
        raise SystemExit(f"--source (checkpoint dir) is required for {name}")
    return cfg, load_hf(cfg, source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen38-27b",
                    choices=["tiny", "tiny-agent", "qwen38-27b", "qwen36-27b"])
    ap.add_argument("--source", default="", help="checkpoint dir; unused by tiny")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    cfg, model = _build(a.model, a.source, a.seed)
    want = PagedKvPool.blocks_for_tokens(a.prompt_tokens)
    eng = build_engine(cfg, model, backend, num_blocks=want + 2, num_slots=2, max_batch=1,
                       max_total_tokens=a.prompt_tokens + 2, prefix_store=NoPrefixStore(),
                       decode_graph=False)
    gen = torch.Generator().manual_seed(a.seed)
    prompt = torch.randint(0, cfg.vocab_size, (a.prompt_tokens,), generator=gen).tolist()
    rid = eng.submit(prompt, SamplingParams(max_new_tokens=1, temperature=0.0))
    req = next(r for r in eng._waiting if r.req_id == rid)

    # Until the row leaves prefill: `max_new_tokens=1` finishes it at the prefill's end, so no
    # decode tick writes KV. Freeing blocks does not clear them, and nothing else allocates.
    ticks = 0
    while req.prefilling and ticks < 4 * want + 8:
        eng.step()
        ticks += 1
    assert req.state_slot is not None, "request was never admitted: the pool is too small"
    assert not req.prefilling, f"prefill did not finish in {ticks} ticks"
    pool, blocks = eng._kv, sorted(req.blocks)
    assert len(blocks) == want, (len(blocks), want)
    tail = a.prompt_tokens % BLOCK_TOKENS or BLOCK_TOKENS

    planes, heads = pool.k_pool.shape[0], pool.num_kv_heads
    print(f"# {a.model}: {planes} full-attn plane(s) x {len(blocks)} blocks x {heads} kv heads, "
          f"block {BLOCK_TOKENS} x head_dim {pool.head_dim} = "
          f"{BLOCK_TOKENS * pool.head_dim} elements per scale")
    print(f"# {a.prompt_tokens} prompt tokens over {ticks} prefill ticks, pool dtype "
          f"{pool.k_pool.dtype}, last block holds {tail} of {BLOCK_TOKENS} tokens")

    out: dict = {"model": a.model, "prompt_tokens": a.prompt_tokens, "ticks": ticks,
                 "planes": planes, "blocks": len(blocks), "kv_heads": heads,
                 "block_tokens": BLOCK_TOKENS, "head_dim": pool.head_dim,
                 "elements_per_scale": BLOCK_TOKENS * pool.head_dim,
                 "pool_dtype": str(pool.k_pool.dtype), "range": {}, "roundtrip": {}}
    sel = {"K": pool.k_pool[:, blocks], "V": pool.v_pool[:, blocks]}

    print("\n# per-(plane, block, head) absmax/absmin_nonzero -- the in-block dynamic range")
    print(f"# {'':<3} {'groups':>7} {'min':>10} {'p50':>10} {'p90':>10} {'p99':>10} "
          f"{'max':>10} {'absmax p50':>11} {'absmax max':>11}")
    for tag, x in sel.items():
        mx, rng = _ranges(x)
        r, m = _dist(rng), _dist(mx)
        out["range"][tag] = {"dynamic_range": r, "absmax": m}
        print(f"  {tag:<3} {r['n']:>7} {r['min']:>10.3e} {r['p50']:>10.3e} {r['p90']:>10.3e} "
              f"{r['p99']:>10.3e} {r['max']:>10.3e} {m['p50']:>11.3e} {m['max']:>11.3e}")

    print("\n# round trip on that same pool data, both scale grids")
    print(f"# {'':<3} {'dtype':>6} {'grid':>11} {'scales':>9} {'err/grp absmax':>15} "
          f"{'|err/x| p50':>12} {'p99':>10} {'max':>10} {'zeroed':>9}")
    for tag, x in sel.items():
        for nick, dt in DTYPES.items():
            for pt in (False, True):
                q = _roundtrip(x, dt, per_token=pt)
                n_sc = x[..., 0].numel() if pt else x[..., 0, 0].numel()
                q["n_scales"] = n_sc
                out["roundtrip"][f"{tag}.{nick}.{q['grid']}"] = q
                print(f"  {tag:<3} {nick:>6} {q['grid']:>11} {n_sc:>9} "
                      f"{q['rel_group_absmax_max']:>15.3e} {q['rel_elem_p50']:>12.3e} "
                      f"{q['rel_elem_p99']:>10.3e} {q['rel_elem_max']:>10.3e} "
                      f"{q['zeroed_frac'] * 100:>8.3f}%")

    # The note's open question. The two columns can disagree, and which wins is measured, not
    # argued: e5m2 spends a mantissa bit, so it is worse per typical element, and buys a grid
    # that reaches 16384x further below a shared scale's absmax, so it can be better on the
    # worst one. A prompt whose blocks are all flat has no worst element for that to help.
    arms = [f"{n}.{g}" for n in DTYPES for g in ("block_head", "per_token")]
    worst = {a_: max(out["roundtrip"][f"{t}.{a_}"]["rel_elem_max"] for t in sel) for a_ in arms}
    typ = {a_: max(out["roundtrip"][f"{t}.{a_}"]["rel_elem_p50"] for t in sel) for a_ in arms}
    out["verdict"] = {"worst_rel_elem": worst, "typical_rel_elem": typ}
    lo, hi = min(worst, key=worst.get), max(worst, key=worst.get)
    tlo, thi = min(typ, key=typ.get), max(typ, key=typ.get)
    print(f"\n# worst element: best {lo} {worst[lo]:.3e}, worst {hi} {worst[hi]:.3e} "
          f"({worst[hi] / worst[lo]:.2f}x); typical: best {tlo} {typ[tlo]:.3e}, "
          f"worst {thi} {typ[thi]:.3e} ({typ[thi] / typ[tlo]:.2f}x)")
    # does the coarse grid's 16x-smaller scale plane cost anything? one line per dtype, on the
    # worst element. per_token is what shipped -- the writer-launch shape forces it -- so this
    # is the record of what block_head would have cost, not a decision input.
    for n in DTYPES:
        b, t_ = worst[f"{n}.block_head"], worst[f"{n}.per_token"]
        print(f"# {n}: shipped per-token grid holds the worst element at {t_:.3e}; block_head "
              f"would be {b:.3e} ({b / t_:.2f}x) at 1/16 the scale memory")

    print()
    print(json.dumps(out, sort_keys=True), flush=True)
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
