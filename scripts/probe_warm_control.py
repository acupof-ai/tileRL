#!/usr/bin/env python3
"""Subprocess-per-cell controls for the warm-vs-cold card divergence (#564).

One 27B engine per process (building several in one process OOMs a 95 GiB
card). Cells, each writes one json:

  coldwave --tag A|B : one COLD engine, B followers in one B=8 wave
  warm1              : one warm engine: publish, then 8 followers SEQUENTIALLY
                       (B=1 each), first-token logits captured per follower
  cold1              : one COLD engine, the same 8 followers sequentially,
                       first-token logits captured
  compare            : read the four jsons, cold-A vs cold-B and warm1 vs cold1

(a) cold waves unequal => a B=8 sparse spec wave is nondeterministic on the card.
(b) B=1 warm != cold   => the warm restore differs on device.
B=1 equal but the B=8 gate diverging => the sm90 B>1 packed-prefill path.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tilerl_kernels.backend import get_backend

from tilerl.build import build_engine
from tilerl.config import qwen38_27b
from tilerl.engine import SamplingParams
from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore
from tilerl.model import load_hf

PROMPT_PAGES = 24
PREFIX_TOKENS = PROMPT_PAGES * BLOCK_TOKENS
TAIL_TOKENS = 20


def _ids(n: int, seed: int, vocab: int) -> list:
    return np.random.default_rng(seed).integers(100, min(vocab, 32000), n).tolist()


def _drain(engine, rid: int, n: int):
    out = []
    for _ in range(n * 8):
        d = engine.poll()
        out.extend(d.get(rid, ()))
        if len(out) >= n:
            break
        engine.step()
    return out[:n]


def _drain_wave(engine, rids: list[int], n: int):
    tok = {r: [] for r in rids}
    for _ in range(n * 8):
        d = engine.poll()
        for r in rids:
            tok[r].extend(d.get(r, ()))
        if all(len(tok[r]) >= n for r in rids):
            break
        engine.step()
    return tok


def _capture_first_logits(eng) -> dict[int, object]:
    first: dict[int, object] = {}
    orig = eng._sample_commit

    def wrap(rows):
        for r, lg, _pos in rows:
            first.setdefault(r.req_id, lg.detach().float().cpu())
        return orig(rows)

    eng._sample_commit = wrap
    return first


def _build(cfg, model, backend, draft, args, prefix: bool):
    return build_engine(
        cfg, model, backend,
        num_blocks=args.num_blocks, num_slots=args.batch + 2,
        max_batch=args.batch + 2, max_total_tokens=8192,
        max_num_batched_tokens=512, sparse_k=args.sparse_k, scorer="bounds",
        kv_cold_bytes=1 << 30, draft=draft, spec_depth=1,
        prefix_store=NoPrefixStore() if not prefix else None)


def cell_coldwave(cfg, model, backend, draft, args, tag: str) -> None:
    eng = _build(cfg, model, backend, draft, args, prefix=False)
    followers = _followers(cfg, args.batch)
    rids = [eng.submit(f, _params(args.steps)) for f in followers]
    tok = _drain_wave(eng, rids, args.steps)
    _write(args.out, {"cell": f"coldwave{tag}",
                      "tokens": [tok[r][:args.steps] for r in rids]})
    eng.shutdown()


def cell_warm1(cfg, model, backend, draft, args) -> None:
    eng = _build(cfg, model, backend, draft, args, prefix=True)
    prompt = _ids(PREFIX_TOKENS, 7, cfg.vocab_size)
    pub = eng.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=256, seed=0))
    _drain(eng, pub, 256)
    entry = eng._sparse.prefix.lookup(prompt)
    assert entry is not None and len(entry["keys"]) == PROMPT_PAGES
    first = _capture_first_logits(eng)
    rows = []
    for i, f in enumerate(_followers(cfg, args.batch)):
        rid = eng.submit(f, _params(args.steps))
        eng.step()
        matched = next(x for x in eng._running if x.req_id == rid).sparse_matched
        toks = _drain(eng, rid, args.steps)
        lg = first.get(rid)
        rows.append({"row": i, "matched": matched, "tokens": toks,
                     "first_tok": toks[0] if toks else None,
                     "logits_sum": float(lg.sum()) if lg is not None else None,
                     "logits_absmean": float(lg.abs().mean()) if lg is not None else None,
                     # full vector saved for compare within this run's json
                     "logits": (lg.numpy().tolist() if lg is not None else None)})
    _write(args.out, {"cell": "warm1", "rows": rows})
    eng.shutdown()


def cell_cold1(cfg, model, backend, draft, args) -> None:
    eng = _build(cfg, model, backend, draft, args, prefix=False)
    first = _capture_first_logits(eng)
    rows = []
    for i, f in enumerate(_followers(cfg, args.batch)):
        rid = eng.submit(f, _params(args.steps))
        toks = _drain(eng, rid, args.steps)
        lg = first.get(rid)
        rows.append({"row": i, "tokens": toks,
                     "first_tok": toks[0] if toks else None,
                     "logits": (lg.numpy().tolist() if lg is not None else None)})
    _write(args.out, {"cell": "cold1", "rows": rows})
    eng.shutdown()


def _followers(cfg, batch: int) -> list[list[int]]:
    prompt = _ids(PREFIX_TOKENS, 7, cfg.vocab_size)
    return [prompt + _ids(TAIL_TOKENS, 100 + i, cfg.vocab_size) for i in range(batch)]


def _params(steps: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_new_tokens=steps, seed=0)


def _write(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"WARM_CONTROL_CELL {obj['cell']} -> {path}", flush=True)


def cell_compare(steps: int, paths: dict[str, str]) -> None:
    data = {}
    for k, v in paths.items():
        with open(v) as fh:
            data[k] = json.load(fh)
    cold_eq = [a == b for a, b in zip(data["A"]["tokens"], data["B"]["tokens"])]
    b1 = []
    for w, c in zip(data["warm1"]["rows"], data["cold1"]["rows"]):
        wt, ct = w["tokens"][:steps], c["tokens"][:steps]
        first_diff = next((j for j in range(min(len(wt), len(ct))) if wt[j] != ct[j]), None)
        max_abs = None
        if w["logits"] is not None and c["logits"] is not None:
            n = min(len(w["logits"]), len(c["logits"]))
            wa = np.asarray(w["logits"][:n], dtype=np.float64)
            ca = np.asarray(c["logits"][:n], dtype=np.float64)
            max_abs = float(np.abs(wa - ca).max())
        b1.append({"row": w["row"], "tokens_equal": wt == ct,
                   "first_diff": first_diff,
                   "first_tok_warm": wt[0] if wt else None,
                   "first_tok_cold": ct[0] if ct else None,
                   "first_logits_max_abs": max_abs})
    result = {"cold_vs_cold_b8_n_equal": sum(cold_eq),
              "cold_vs_cold_b8": cold_eq,
              "b1_warm_vs_cold": b1,
              "b1_n_equal": sum(x["tokens_equal"] for x in b1)}
    print("WARM_CONTROL_RESULT " + json.dumps(result))
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True,
                   choices=["coldwave", "warm1", "cold1", "compare"])
    p.add_argument("--tag", default="")
    p.add_argument("--source", default="/work/tilerl-ckpt/Qwen3.8-27B-NVFP4")
    p.add_argument("--draft",
                   default="/work/tilerl-ckpt/Qwen3.8-27B-NVFP4/model_mtp.safetensors")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sparse-k", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=512)
    p.add_argument("--out", default="/work/warmctl_cell.json")
    p.add_argument("--a", default="/work/warmctl_A.json")
    p.add_argument("--b", default="/work/warmctl_B.json")
    p.add_argument("--w", default="/work/warmctl_warm1.json")
    p.add_argument("--c", default="/work/warmctl_cold1.json")
    args = p.parse_args()

    if args.cell == "compare":
        cell_compare(args.steps, {"A": args.a, "B": args.b,
                                  "warm1": args.w, "cold1": args.c})
        return

    from tilerl.spec import load_draft

    cfg = qwen38_27b()
    model = load_hf(cfg, args.source)
    backend = get_backend()
    draft = load_draft(model, args.draft)
    if args.cell == "coldwave":
        cell_coldwave(cfg, model, backend, draft, args, args.tag)
    elif args.cell == "warm1":
        cell_warm1(cfg, model, backend, draft, args)
    else:
        cell_cold1(cfg, model, backend, draft, args)


if __name__ == "__main__":
    main()
