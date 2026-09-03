"""Every toggle that can be turned on, measured against one reference arm — in
one process on one card.

Extends ``acc_spec_arms.py``. What that script did for one toggle (speculation),
this does for a list of them: N arms in ONE process, each differing from arm 0
by a single switch, greedy ``temperature=0.0 seed=0``, and a per-question
comparison of the TOKEN IDS rather than of the score.

Three numbers per arm, all three or the arm is not done:

1. GSM8K greedy accuracy (generative — the only suite here that decodes).
2. Token identity against the reference arm: how many completions differ, and
   for each, the reference arm's top-1/top-2 logit gap at the first differing
   generated index. A flip at a 1e-6 gap is arithmetic; at 0.5 it is a bug.
3. Wall-clock throughput and peak CUDA memory, end to end, same session. Never
   a product of a tick cost and an acceptance rate.

MMLU is a NEGATIVE CONTROL, not an accuracy gate: ``mmlu_score`` runs
``max_new_tokens=1``, so the letter comes off the prefill and decode never
happens. Speculation cannot fire; the arms must be bit-identical. If they are
not, something other than the toggle under test is wrong.

The prompt set is designed, not sampled: one GSM8K question per residue of
prompt length mod 64, so all 64 residues of the KV length the decode kernel
sees are entered, not just the handful a random slice reaches.

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src:packages/tilerl-kernels/src \
    TILERL_TARGET=cuda /work/tl013/bin/python scripts/toggle_matrix.py \
        --source /work/Qwen3.8-27B-NVFP4 \
        --draft /work/Qwen3.8-27B-NVFP4/model_mtp.safetensors \
        --gsm8k /work/gsm8k_test.jsonl --group engine --out /work/mtx-engine
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

import tilelang
from tilerl.config import qwen38_27b
from tilerl.engine import build_engine
from tilerl.eval import answer_match, letter, mmlu_questions
from tilerl.kv_cache import NoPrefixStore
from tilerl.model import load_hf
from tilerl.prompt import render_chat, sampling
from tilerl.tokenizer import get_tokenizer
from tilerl_kernels import backend as backend_mod
from tilerl_kernels.backend import get_backend

RES = 64  # KV block residue class: BLOCK_TOKENS=16, the split-KV tile is 64


# --------------------------------------------------------------- the matrix


@dataclass
class Arm:
    """One row of the matrix: the reference arm plus exactly one switch."""

    name: str
    switch: str                                   # what this arm flips, for the report
    engine: dict = field(default_factory=dict)    # build_engine overrides
    env: dict = field(default_factory=dict)       # call-time env reads
    patch: dict = field(default_factory=dict)     # module attributes (import-time reads)
    concurrency: int = 8
    fused: bool = True
    tf32: bool | None = None
    num_blocks: int | None = None


def groups(draft_path: str) -> dict[str, list[Arm]]:
    """Arms grouped by what they can share a process with. An import-time env
    read cannot be flipped in-process, so it is patched at its module attribute
    (`--check-patch` proves the patch reaches the dispatch) or split out."""
    ref = Arm("ref", "reference: no draft, no graph, no prefix store")
    return {
        # Engine seam: one card, one process, six switches off the same weights.
        "engine": [
            ref,
            Arm("graph", "decode_graph=True", engine=dict(decode_graph=True)),
            Arm("prefix", "prefix_store=PrefixStore", engine=dict(prefix_store="real")),
            Arm("spec-w2", "draft + spec_depth=1 (W=2)", engine=dict(draft=draft_path, width=2)),
            Arm("spec-w4", "draft + spec_depth=3 (W=4)", engine=dict(draft=draft_path, width=4)),
            Arm("spec-w8", "draft + spec_depth=7 (W=8)", engine=dict(draft=draft_path, width=8)),
            Arm("spec-w8-graph", "W=8 + decode_graph=True",
                engine=dict(draft=draft_path, width=8, decode_graph=True)),
            # _MAX_VERIFY_W=8: at W=12 the verify tick leaves the decode
            # attention kernel for the M-tiled prefill one. Legal (spec_depth <
            # BLOCK_TOKENS) and, per the W-verify entry, uncovered by parity.
            Arm("spec-w12", "W=12 — past _MAX_VERIFY_W, onto the M-tiled kernel",
                engine=dict(draft=draft_path, width=12)),
            # The KV-split selector is host-static: ks=64 needs the pool past
            # 65536 tokens (4096 blocks) or a narrow enough grid.
            Arm("kvsplit-64", "num_blocks=4608 -> KVSPLIT=64 attention cells",
                num_blocks=4608),
        ],
        # Kernel / numeric switches read at call time, plus the two read at
        # import time that a module attribute can still reach.
        "kernel": [
            ref,
            Arm("gdn-chunk64", "TILERL_GDN_CHUNKWISE=64 (prefill chunkwise-WY reference)",
                env={"TILERL_GDN_CHUNKWISE": "64"}),
            Arm("gdn-fla", "TILERL_GDN_CHUNKWISE=64 + TILERL_GDN_FLA=1",
                env={"TILERL_GDN_CHUNKWISE": "64", "TILERL_GDN_FLA": "1"}),
            Arm("tf32-off", "TILERL_TF32=0 (torch.backends tf32 off)", tf32=False),
        ],
        # fuse_projections is a load-time switch, so its arm holds a SECOND 27B
        # resident; kept out of the other groups so their peak-memory rows stay
        # the serving figure.
        "fuse": [ref, Arm("fuse-off", "load_hf(fuse_projections=False)", fused=False)],
        # TILERL_RED_TILE is read at import into kernels_linear._RED_TILE, which
        # the JIT bakes into the kernel, AND copied into backend._MMA_RED, which
        # pads the operands. Patching one without the other truncates the
        # reduction (errors/2026-09-03-red-tile-zeroed-the-weight-gradient.md),
        # so this group is one arm per PROCESS, same card, back to back.
        "redtile": [ref],
        # _MGEMV only fires at 2 <= M <= 3, so it has to be measured at a batch
        # that produces those rows. B=2 with W=1 is M=2.
        "mgemv": [
            replace(ref, name="ref-b2", concurrency=2),
            Arm("mgemv-off-b2", "TILERL_MGEMV=0 (backend._MGEMV)",
                patch={"backend._MGEMV": 0}, concurrency=2),
            replace(ref, name="ref-b1", concurrency=1),
            Arm("spec-w2-b1", "W=2 at B=1 (M=2, the MGEMV regime)",
                engine=dict(draft=draft_path, width=2), concurrency=1),
            Arm("mgemv-off-spec-w2-b1", "W=2 at B=1 with TILERL_MGEMV=0",
                engine=dict(draft=draft_path, width=2), patch={"backend._MGEMV": 0},
                concurrency=1),
        ],
        # Batch is a matrix axis too: the linear plan picks a different kernel
        # per M regime, so B can move the tokens as well as the throughput.
        "batch": [
            replace(ref, name="ref-b8"),
            replace(ref, name="ref-b1", concurrency=1),
            replace(ref, name="ref-b4", concurrency=4),
        ],
    }


# --------------------------------------------------------------- the prompts


def designed_prompts(tok, rows: list[dict], want: int = RES) -> list[tuple[dict, str, list[int]]]:
    """One question per residue of prompt length mod 64, ``want`` residues of
    them. A random slice reaches a handful of residues; this reaches all of the
    ones it claims, and the residue count is what makes the run a gate rather
    than an anecdote."""

    def enc(text: str) -> list[int]:
        return tok.encode(render_chat([("user", text)], False))

    picks: dict[int, tuple[dict, str, list[int]]] = {}
    for r in rows:
        ids = enc(r["prompt"])
        res = len(ids) % RES
        if res < want:
            picks.setdefault(res, (r, r["prompt"], ids))
        if len(picks) == want:
            break
    base = rows[0]
    for res in range(want):  # pad a base question until it lands on a missing residue
        if res in picks:
            continue
        for j in range(1, 4 * RES):
            text = ("ok " * j) + base["prompt"]
            ids = enc(text)
            if len(ids) % RES == res:
                picks[res] = (base, text, ids)
                break
    return [picks[k] for k in sorted(picks)]


# --------------------------------------------------------------- the tracer


class Trace:
    """Sync-free per-tick instrumentation. Every host read of a device tensor is
    deferred to :meth:`resolve`: a ``bool(t[i])`` inside the tick would drain the
    pipeline once per row and the throughput row would then be measuring the
    tracer. Residues come from ``r.seq_len``, a python int, so they are free."""

    def __init__(self, engine):
        self.res, self.res_wide, self.widths = Counter(), Counter(), Counter()
        self.nan_keys: list[list[tuple[int, int]]] = []
        self.nan_flags: list[torch.Tensor] = []
        self.gap_keys: list[list[tuple[int, int]]] = []
        self.gap_vals: list[torch.Tensor] = []
        self._e = engine
        self._fwd, self._verify, self._sample = (
            engine._run_forward, engine._verify, engine._sample_batch)
        engine._run_forward = self._traced_fwd
        engine._verify = self._traced_verify
        engine._sample_batch = self._traced_sample

    def _traced_fwd(self, decodes, prefills, chunks):
        if decodes:
            w = max((1 + len(r.drafts) for r in decodes), default=1)
            self.widths[w] += 1
            for r in decodes:
                self.res[(r.seq_len - 1 + w) % RES] += 1
                if w > 1:
                    self.res_wide[(r.seq_len - 1 + w) % RES] += 1
        return self._fwd(decodes, prefills, chunks)

    def _traced_verify(self, rows, chains, logits, hidden):
        self.nan_keys.append([(r.req_id, r.seq_len - 1 + len(chains[i]))
                              for i, r in enumerate(rows)])
        self.nan_flags.append(torch.isnan(logits).flatten(1).any(dim=1))
        return self._verify(rows, chains, logits, hidden)

    def _traced_sample(self, rows):
        if rows:
            top = torch.stack([l for _, l, _ in rows]).float().topk(2, dim=-1).values
            self.gap_keys.append([(r.req_id, g) for r, _, g in rows])
            self.gap_vals.append(top[:, 0] - top[:, 1])
        return self._sample(rows)

    def detach(self) -> None:
        self._e._run_forward, self._e._verify, self._e._sample_batch = (
            self._fwd, self._verify, self._sample)

    def resolve(self) -> tuple[dict, dict]:
        """(summary, gap by (req_id, generated index)). One host read, at the end."""
        nan = Counter()
        if self.nan_flags:
            keys = [k for b in self.nan_keys for k in b]
            for (_, n), bad in zip(keys, torch.cat(self.nan_flags).tolist()):
                if bad:
                    nan[n % RES] += 1
        gaps: dict[tuple[int, int], float] = {}
        if self.gap_vals:
            keys = [k for b in self.gap_keys for k in b]
            for k, v in zip(keys, torch.cat(self.gap_vals).tolist()):
                gaps[k] = v  # a rejected draft position is re-sampled; the commit wins
        return (dict(res=dict(self.res), res_wide=dict(self.res_wide),
                     widths=dict(self.widths), nan=dict(nan),
                     residues_reached=len(self.res), wide_residues_reached=len(self.res_wide),
                     nan_row_ticks=sum(nan.values())), gaps)


# --------------------------------------------------------------- one arm


def generate_ids(engine, prompt_ids: list[list[int]], sp, concurrency: int):
    """``eval.generate`` keeping the token ids and the request id — a string
    comparison cannot say WHERE two arms diverged, and the gap lookup is keyed
    by (request, generated index)."""
    out: list = [None] * len(prompt_ids)
    rids: list = [None] * len(prompt_ids)
    pending, todo = {}, list(enumerate(prompt_ids))
    while pending or todo:
        while todo and len(pending) < concurrency:
            i, ids = todo.pop()
            rid = engine.submit(ids, sp)
            pending[rid], rids[i] = i, rid
        engine.step()
        for wid, ids in engine.poll().items():
            out[pending.pop(wid)] = ids
    return out, rids


def phase(engine, fn):
    torch.cuda.synchronize()
    s0, t0 = engine.stats(), time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    secs, s1 = time.perf_counter() - t0, engine.stats()
    d = {k: s1[k] - s0[k] for k in ("tokens_generated", "decode_forwards", "prefill_forwards",
                                    "mixed_forwards", "spec_drafted", "spec_accepted")}
    d["secs"] = secs
    d["tok_per_s"] = d["tokens_generated"] / secs
    d["tok_per_decode_forward"] = (d["tokens_generated"] / d["decode_forwards"]
                                   if d["decode_forwards"] else 0.0)
    d["accept_rate"] = (d["spec_accepted"] / d["spec_drafted"]) if d["spec_drafted"] else 0.0
    return result, d


class Patched:
    """Env vars, module attributes and the tf32 backend flag, scoped to one arm."""

    def __init__(self, a: Arm):
        self.a, self.old_env, self.old_attr, self.old_tf32 = a, {}, {}, None

    def __enter__(self):
        for k, v in self.a.env.items():
            self.old_env[k] = os.environ.get(k)
            os.environ[k] = v
        for path, v in self.a.patch.items():
            mod, attr = path.split(".")
            m = {"backend": backend_mod}[mod]
            self.old_attr[path] = getattr(m, attr)
            setattr(m, attr, v)
        if self.a.tf32 is not None:
            self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = self.a.tf32
            torch.backends.cudnn.allow_tf32 = self.a.tf32
        return self

    def __exit__(self, *_):
        for k, v in self.old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        for path, v in self.old_attr.items():
            setattr({"backend": backend_mod}[path.split(".")[0]], path.split(".")[1], v)
        if self.old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self.old_tf32
            torch.backends.cudnn.allow_tf32 = self.old_tf32


def run_arm(a: Arm, cfg, models, backend, tok, prompts, sp, mmlu, num_blocks, seen) -> dict:
    from tilerl.spec import load_draft

    e = dict(a.engine)
    width = e.pop("width", 1)
    draft_path = e.pop("draft", None)
    if e.get("prefix_store") == "real":
        e["prefix_store"] = None  # build_engine's own default is PrefixStore
    out = {"arm": a.name, "switch": a.switch, "width": width, "concurrency": a.concurrency,
           "fused": a.fused, "engine_kw": {k: str(v) for k, v in e.items()}}

    with Patched(a):
        model = models[a.fused]
        draft = load_draft(model, draft_path) if draft_path else None
        # +1 slot: with decode_graph on, the first under-filled bucket takes a
        # pad slot and a pad block and never gives them back (engine.py:678),
        # so num_slots == concurrency admits one request fewer and submit
        # raises "LinearStatePool exhausted" instead of queueing.
        kw = dict(num_blocks=a.num_blocks or num_blocks, num_slots=a.concurrency + 1,
                  max_batch=8, decode_graph=False,
                  prefix_store=NoPrefixStore(), draft=draft, spec_depth=max(1, width - 1))
        kw.update(e)
        engine = build_engine(cfg, model, backend, **kw)
        out["prefix_store_cls"] = type(engine._prefix).__name__
        out["mgemv"] = backend_mod._MGEMV
        out["mma_red"] = backend_mod._MMA_RED
        out["gdn_chunkwise"] = os.environ.get("TILERL_GDN_CHUNKWISE", "0")
        out["gdn_fla"] = os.environ.get("TILERL_GDN_FLA", "")
        out["tf32"] = torch.backends.cuda.matmul.allow_tf32

        # Warm the JIT and the graph capture at every batch the phase will see.
        # A uniform warmup finishes all rows on one tick and never compiles the
        # tail shapes; the linear plan then JITs linear_fp4_gemv[2] mid-timing
        # and the throughput row measures the compiler (6.6 tok/s on the smoke).
        warm = [p[2] for p in prompts[:a.concurrency]]
        for wid, ids in enumerate(warm):
            engine.submit(ids, replace(sp, max_new_tokens=8 + 4 * wid))
        while engine.stats()["running"] or engine.stats()["waiting"]:
            engine.step()
            engine.poll()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        tr = Trace(engine)
        if mmlu is not None:  # negative control: max_new_tokens=1, decode never runs
            (ids, rids), d = phase(engine, lambda: generate_ids(
                engine, mmlu["ids"], mmlu["sp"], a.concurrency))
            preds = [letter(tok.decode(t)) for t in ids]
            out["mmlu"] = dict(correct=sum(p == g for p, g in zip(preds, mmlu["gold"])),
                               total=len(preds), timing=d, tokens=ids, rids=rids, pred=preds)
            print(f"[{a.name}] mmlu CONTROL {out['mmlu']['correct']}/{len(preds)} "
                  f"decode_forwards {d['decode_forwards']} (0 expected)", flush=True)

        warm_keys = {f"{k[0]}{list(k[1]) if k[1] else ''}" for k in backend._kernels}
        (ids, rids), d = phase(engine, lambda: generate_ids(
            engine, [p[2] for p in prompts], sp, a.concurrency))
        ok = [answer_match(tok.decode(t), r["answer"]) for t, (r, _, _) in zip(ids, prompts)]
        out["gsm8k"] = dict(correct=sum(ok), total=len(prompts), timing=d, tokens=ids,
                            rids=rids, prompt_tokens=[len(p[2]) for p in prompts],
                            out_tokens=[len(t) for t in ids])
        tr.detach()
        summary, gaps = tr.resolve()
        out["trace"] = summary
        out["gaps"] = {f"{k[0]}:{k[1]}": v for k, v in gaps.items()}
        out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out["prefix_snapshots"] = len(getattr(engine, "_prefix_state", {}))
        # Which cells this arm actually compiled. This is the switch's negative
        # control: a patched _MGEMV that never reaches the dispatch would leave
        # ('linear_fp4_gemv', M) in the cache anyway, and the arm would be a
        # relabelled copy of the reference.
        keys = {f"{k[0]}{list(k[1]) if k[1] else ''}" for k in backend._kernels}
        out["kernels_new"] = sorted(keys - seen)
        out["jit_during_timing"] = sorted(keys - warm_keys)  # non-empty ⇒ the row is compiler
        seen |= keys

        print(f"[{a.name}] gsm8k {sum(ok)}/{len(prompts)}  {d['secs']:.1f}s  "
              f"{d['tok_per_s']:.1f} tok/s  {d['tok_per_decode_forward']:.2f} tok/decode-fwd  "
              f"accept {100 * d['accept_rate']:.1f}%  peak {out['peak_gib']:.2f} GiB\n"
              f"[{a.name}] residues {summary['residues_reached']}/64 "
              f"(wide {summary['wide_residues_reached']}/64)  widths {summary['widths']}  "
              f"NaN row-ticks {summary['nan_row_ticks']}"
              + (f"\n[{a.name}] WARNING: JIT inside the timed phase, throughput is not a "
                 f"throughput: {out['jit_during_timing']}" if out["jit_during_timing"] else ""),
              flush=True)

    engine = draft = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


# --------------------------------------------------------------- the report


def compare(ref: dict, arm: dict, suite: str) -> dict:
    """Token-for-token, against the reference arm. The gap is the REFERENCE
    arm's top-1/top-2 margin at the first differing generated index: a flip at
    1e-6 is arithmetic, at 0.5 it is a bug."""
    a, b = ref[suite], arm[suite]
    ar, br = a.get("rids") or [], b.get("rids") or []
    diffs = []
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x == y:
            continue
        k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
        diffs.append({"q": i, "index": k, "ref_len": len(x), "arm_len": len(y),
                      "ref_tok": x[k] if k < len(x) else None,
                      "arm_tok": y[k] if k < len(y) else None,
                      "ref_gap": ref["gaps"].get(f"{ar[i]}:{k}") if i < len(ar) else None,
                      "arm_gap": arm["gaps"].get(f"{br[i]}:{k}") if i < len(br) else None})
    gaps = [d["ref_gap"] for d in diffs if d["ref_gap"] is not None]
    bins = Counter()
    for g in gaps:
        bins["<1e-4" if g < 1e-4 else "<1e-2" if g < 1e-2 else
             "<0.1" if g < 0.1 else "<0.5" if g < 0.5 else ">=0.5"] += 1
    return {"differing": len(diffs), "total": a["total"], "first": diffs[:20],
            "gap_bins": dict(bins), "gap_max": max(gaps) if gaps else None,
            "gap_median": sorted(gaps)[len(gaps) // 2] if gaps else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--arms", help="comma-separated subset of the group")
    p.add_argument("--mmlu-n", type=int, default=0, help="0 skips the control")
    p.add_argument("--prompts", type=int, default=RES)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.getLogger("TileLang").setLevel(logging.WARNING)  # one INFO pair per JIT'd kernel

    backend = get_backend()
    assert backend.device.type == "cuda", "needs TILERL_TARGET=cuda"
    print(f"tilelang {tilelang.__version__}  torch {torch.__version__}  "
          f"arch {backend.arch}  device {torch.cuda.get_device_name(0)}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    cfg = qwen38_27b()
    tok = get_tokenizer(args.source)
    rows = [json.loads(ln) for ln in Path(args.gsm8k).read_text().splitlines() if ln.strip()]
    prompts = designed_prompts(tok, rows, args.prompts)
    lens = [len(p[2]) for p in prompts]
    print(f"prompt set: {len(prompts)} questions, "
          f"{len({n % RES for n in lens})}/{RES} starting residues, "
          f"lengths {min(lens)}-{max(lens)}", flush=True)

    arms = groups(args.draft)[args.group]
    if args.arms:
        keep = set(args.arms.split(","))
        arms = [a for a in arms if a.name in keep]
    models = {True: load_hf(cfg, args.source, fuse_projections=True)}
    if any(not a.fused for a in arms):
        models[False] = load_hf(cfg, args.source, fuse_projections=False)

    sp = sampling(tok, False, args.max_new_tokens, temperature=0.0, max_think_tokens=0, seed=0)
    mmlu = None
    if args.mmlu_n:
        from tilerl.engine import SamplingParams
        mp, gold, _ = mmlu_questions(args.mmlu_n, seed=0)
        allowed = tuple(sorted({tok.encode(f" {c}")[-1] for c in "ABCD"}
                               | {tok.encode(c)[-1] for c in "ABCD"}))
        mmlu = {"ids": [tok.encode(x) for x in mp], "gold": gold,
                "sp": SamplingParams(temperature=0.0, max_new_tokens=1, seed=0,
                                     allowed_ids=allowed)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done, seen = [], set()
    for a in arms:
        r = run_arm(a, cfg, models, backend, tok, prompts, sp, mmlu, args.num_blocks, seen)
        (out / f"{r['arm']}.json").write_text(json.dumps(r))
        done.append(r)

    ref = done[0]
    report = {"group": args.group, "reference": ref["arm"], "arms": []}
    print(f"\n=== matrix: every arm against '{ref['arm']}' ===")
    for r in done:
        row = {"arm": r["arm"], "switch": r["switch"],
               "gsm8k": f"{r['gsm8k']['correct']}/{r['gsm8k']['total']}",
               "tok_per_s": round(r["gsm8k"]["timing"]["tok_per_s"], 1),
               "tok_per_decode_fwd": round(r["gsm8k"]["timing"]["tok_per_decode_forward"], 2),
               "accept": round(100 * r["gsm8k"]["timing"]["accept_rate"], 1),
               "peak_gib": round(r["peak_gib"], 2),
               "residues": f"{r['trace']['residues_reached']}/64",
               "wide_residues": f"{r['trace']['wide_residues_reached']}/64",
               "nan_row_ticks": r["trace"]["nan_row_ticks"],
               "widths": r["trace"]["widths"], "kernels_new": r["kernels_new"],
               "jit_during_timing": r["jit_during_timing"],
               "mgemv": r["mgemv"], "mma_red": r["mma_red"], "tf32": r["tf32"],
               "gdn_chunkwise": r["gdn_chunkwise"], "gdn_fla": r["gdn_fla"]}
        if r is not ref:
            row["gsm8k_vs_ref"] = compare(ref, r, "gsm8k")
            if mmlu is not None:
                row["mmlu_control_vs_ref"] = compare(ref, r, "mmlu")
        report["arms"].append(row)
        print(json.dumps(row, default=str)[:600], flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}/report.json")


if __name__ == "__main__":
    main()
