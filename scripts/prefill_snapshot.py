"""PROBE-ONLY #298892: cross-process prefill snapshots for the sparse engine.

A completed prompt's published prefix entry lives in
``SparsePrefixCache`` and names one shared host blob per page (KV + Quest
bounds, plus the draft head's ``dk`` when a draft is loaded) and one GDN
``(states, conv_window)`` snapshot at the page boundary (and the optional trunk
``hidden`` a spec follower's first tail token conditions on). This tool dumps
that frozen entry to a directory and, in a SECOND process, re-registers the
blobs in the cold tier and puts the entry back on the cache's lookup chains.
``submit`` then takes the production prefix-hit adoption path unchanged — no
page is pre-allocated: a named page promotes lazily into a private fresh block
when the selector asks for it.

Nothing under ``src/`` is modified.

Snapshot identity (``meta.json``) must cover everything that changes a prefill
result: the prompt-token hash, a caller-supplied model key, and the sparse
geometry/draft config read off the engine. A mismatch raises instead of
silently recomputing. Page blobs are stored under their content keys but the
keys are NOT re-hashed over the bytes on load (the tier trusts them), so the
``__main__`` gate tampers a page to prove the restored K/V is what is read.

Layout: ``meta.json`` + ``state.pt`` + ``page_<key>.pt`` per page.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

from tilerl import sparse_index
from tilerl.kv_cache import BLOCK_TOKENS

SNAPSHOT_VERSION = 1


def _token_hash(tokens) -> str:
    h = hashlib.sha256()
    h.update(len(tokens).to_bytes(8, "little"))
    h.update(torch.as_tensor(list(tokens), dtype=torch.long).numpy().tobytes())
    return h.hexdigest()


def config_fingerprint(e, model_key: str) -> dict:
    """The engine knobs that change a prefill K/V result. Read, not guessed:
    sparse_k/scorer pick pages, min-tokens gates sparse vs dense prefill, the
    window geometry bounds the own span, and the draft window shapes the stored
    draft blob."""
    draft = getattr(e, "_draft", None)
    return {
        "version": SNAPSHOT_VERSION,
        "block_tokens": BLOCK_TOKENS,
        "sparse_k": int(e._sparse.tracker.k_pages),
        "scorer": str(e._sparse.tracker.scorer),
        "sparse_min_tokens": int(e._sparse_min_tokens),
        "window_pages": int(sparse_index.WINDOW_PAGES),
        "draft_attn_window_tokens": (None if draft is None else int(draft.attn_window_tokens)),
        "model_key": str(model_key),
    }


def dump(e, tokens, out_dir: str, model_key: str = "") -> str:
    """Dump the FROZEN prompt-prefix entry ``tokens`` currently resolves to.
    Call after the publisher's prefill completed (the prompt-end entry exists)."""
    sp = e._sparse
    if sp is None or sp.prefix is None:
        raise RuntimeError("snapshot dump needs a sparse engine with a prefix cache")
    entry = sp.prefix.lookup(list(tokens))
    if entry is None:
        raise RuntimeError("no published prefix entry for the prompt; run it first")
    if tuple(entry["tokens"]) != tuple(int(t) for t in tokens)[: len(entry["tokens"])]:
        # A shorter/partial hit would snapshot a different prompt than asked.
        raise RuntimeError("entry is a partial-prefix hit; dump the exact prompt length")
    cold = e._kv.cold
    os.makedirs(out_dir, exist_ok=True)
    pages = []
    for p, key in enumerate(entry["keys"]):
        blob = cold.share_take(key)
        if blob is None:
            raise RuntimeError(f"page {p}: content key {key} has no shared blob")
        path = os.path.join(out_dir, f"page_{key}.pt")
        torch.save(
            {k: v.detach().cpu().clone() for k, v in blob.items() if torch.is_tensor(v)}, path
        )
        pages.append(
            {
                "page": p,
                "key": int(key),
                "fields": sorted(k for k, v in blob.items() if torch.is_tensor(v)),
            }
        )
    states, window = entry["state"]
    torch.save(
        {
            "states": states.detach().cpu().clone(),
            "window": None if window is None else window.detach().cpu().clone(),
            "hidden": (
                None if entry.get("hidden") is None else entry["hidden"].detach().cpu().clone()
            ),
        },
        os.path.join(out_dir, "state.pt"),
    )
    meta = {
        "n_tokens": len(entry["tokens"]),
        "tokens": [int(t) for t in entry["tokens"]],
        "token_hash": _token_hash(entry["tokens"]),
        "entry_hash": int(entry["hash"]),
        "config": config_fingerprint(e, model_key),
        "pages": pages,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return out_dir


def load(e, snap_dir: str, tokens, model_key: str = "") -> dict:
    """Restore a snapshot into a FRESH engine's empty sparse prefix cache.
    Refuses (raises) on any identity/config/token mismatch; never recomputes."""
    with open(os.path.join(snap_dir, "meta.json")) as f:
        meta = json.load(f)
    want_cfg = config_fingerprint(e, model_key)
    if meta["config"] != want_cfg:
        raise ValueError(f"snapshot config mismatch:\n got {meta['config']}\n exp {want_cfg}")
    toks = [int(t) for t in tokens][: meta["n_tokens"]]
    if meta["n_tokens"] != len(toks) or meta["token_hash"] != _token_hash(toks):
        raise ValueError("snapshot token hash/length mismatch; refusing to load")
    sp = e._sparse
    if sp is None or sp.prefix is None:
        raise RuntimeError("snapshot load needs a sparse engine with a prefix cache")
    if sp.prefix.lookup(toks) is not None:
        raise RuntimeError("engine already has a matching entry; load into a fresh cache")
    cold = e._kv.cold
    keys = []
    for rec in meta["pages"]:
        key = rec["key"]
        blob = torch.load(
            os.path.join(snap_dir, f"page_{key}.pt"), map_location="cpu", weights_only=True
        )
        if set(blob) != set(rec["fields"]):
            raise ValueError(f"page {key}: blob fields {sorted(blob)} != manifest {rec['fields']}")
        nbytes = sum(t.numel() * t.element_size() for t in blob.values())
        cold.share_hold(key, blob, nbytes)  # one entry ref, exactly like a frozen copy
        keys.append(key)
    sd = torch.load(os.path.join(snap_dir, "state.pt"), map_location="cpu", weights_only=True)
    state = (sd["states"], sd["window"])
    entry = {
        "eid": sp.prefix._next_id,
        "tokens": tuple(toks),
        "keys": keys,
        "state": state,
        "hash": meta["entry_hash"],
        "hidden": sd["hidden"],
    }
    sp.prefix._next_id += 1
    sp.prefix._entries.setdefault(entry["hash"], []).append(entry)
    sp.prefix._by_id[entry["eid"]] = entry
    return entry


# --------------------------------------------------------------------------- #
# CPU tiny gate: restored snapshot == full-prefill, bit for bit; negatives red.
# --------------------------------------------------------------------------- #
def _build():
    import numpy as np  # noqa: F401

    from tilerl.build import build_engine
    from tilerl.config import tiny
    from tilerl.model import build_random
    from tilerl.testing import RefBackend

    return build_engine(
        cfg=tiny(),
        model=build_random(tiny(), seed=11),
        backend=RefBackend(),
        num_blocks=4096,
        num_slots=4,
        max_batch=1,
        max_total_tokens=65536,
        max_num_batched_tokens=2048,
        sparse_k=128,
        scorer="bounds",
        kv_cold_bytes=1 << 30,
        decode_graph=True,
    )


def _drive(e, prompt, n_new, cap_logits: dict, matched: dict | None = None):
    from tilerl.engine import SamplingParams

    orig = e._sample_batch

    def cap(rows):
        out = orig(rows)
        for r, lg, _gi in rows:
            if id(r) not in cap_logits:
                cap_logits[id(r)] = lg.detach().float().cpu().clone()
        return out

    e._sample_batch = cap
    rid = e.submit(prompt, SamplingParams(temperature=0.0, max_new_tokens=n_new, seed=0))
    out = None
    for _ in range(40000):
        e.step()
        if matched is not None and matched.get("v") is None:
            r = next((x for x in e._running if x.req_id == rid), None)
            if r is not None:
                matched["v"] = int(getattr(r, "sparse_matched", 0))
        out = e.take(rid)
        if out is not None:
            break
    assert out is not None, "request never finished"
    return list(out)


def _selfcheck() -> None:
    import shutil
    import tempfile

    import numpy as np

    pages = 6
    prompt = ((np.arange(pages * BLOCK_TOKENS, dtype=np.int64) % 300) + 7).tolist()
    n_new = 10
    root = tempfile.mkdtemp(prefix="snap_gate_")
    snap = os.path.join(root, "snap")
    try:
        # publisher: prefill + a short decode closes the prompt-end entry
        pub = _build()
        _drive(pub, prompt, 8, {})
        dump(pub, prompt, snap, model_key="cpu-tiny")
        pub.shutdown()

        # reference: full prefill from a MISS (no snapshot loaded)
        ref = _build()
        ref_lg: dict = {}
        ref_out = _drive(ref, prompt, n_new, ref_lg)
        ref_first = next(iter(ref_lg.values()))
        ref.shutdown()

        # positive: snapshot restore must reproduce outputs and first-tick logits
        hit = _build()
        load(hit, snap, prompt, model_key="cpu-tiny")
        hit_lg, hit_matched, hit_out = {}, {"v": None}, None
        hit_out = _drive(hit, prompt, n_new, hit_lg, hit_matched)
        hit.shutdown()
        hit_first = next(iter(hit_lg.values()))
        assert hit_matched["v"] == pages * BLOCK_TOKENS, (
            f"restored snapshot adopted {hit_matched['v']}, expected full prefix"
        )
        assert hit_out == ref_out, "snapshot temp0 outputs differ from full prefill"
        assert torch.equal(hit_first, ref_first), "first-tick logits differ from full prefill"

        # negative 1: tamper page 0's K blob -> gate MUST red. The page is named
        # by its MANIFEST content key (a sorted filename can land on the trailing
        # page, which the aligned adoption re-forwards and never reads back).
        bad = os.path.join(root, "tampered")
        shutil.copytree(snap, bad)
        with open(os.path.join(bad, "meta.json")) as f:
            bmeta = json.load(f)
        bp = os.path.join(bad, f"page_{bmeta['pages'][0]['key']}.pt")
        t = torch.load(bp, map_location="cpu", weights_only=True)
        t["k"].view(-1)[0] += 5.0
        torch.save(t, bp)
        tam = _build()
        load(tam, bad, prompt, model_key="cpu-tiny")
        tam_lg: dict = {}
        tam_out = _drive(tam, prompt, n_new, tam_lg)
        tam.shutdown()
        assert (next(iter(tam_lg.values())) != ref_first).any() and tam_out != ref_out, (
            "tampered page was NOT read (vacuous negative): logits and outputs unchanged"
        )

        # negative 2: fingerprint mismatch -> load refuses, no silent recompute
        wcfg = os.path.join(root, "wrongcfg")
        shutil.copytree(snap, wcfg)
        mp = os.path.join(wcfg, "meta.json")
        with open(mp) as f:
            m = json.load(f)
        m["config"]["sparse_k"] += 1
        with open(mp, "w") as f:
            json.dump(m, f)
        w = _build()
        refused = False
        try:
            load(w, wcfg, prompt, model_key="cpu-tiny")
        except ValueError:
            refused = True
        finally:
            w.shutdown()
        assert refused, "a changed sparse_k config was accepted (fingerprint not enforced)"
        print(
            "PREFILL-SNAPSHOT GATE OK: outputs+first-logits identical; "
            "tamper and fingerprint negatives both red"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    else:
        ap.error("probe module: use --selfcheck or import dump/load")
