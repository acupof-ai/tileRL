"""Acceptance / draft-cost / tok-s sweep over the draft trailing-window W.

Acceptance tool for the #684 gate (TILERL_DRAFT_ATTN_WINDOW_TOKENS /
``draft.attn_window_tokens``). It answers the product question the #668 probe
could not: as the draft decode READ window shrinks from the full prefix to W
tokens, does acceptance hold while the (linear-in-context) draft attention gets
cheaper?

Design, carried over from ab_draft_depth.py:
- ONE engine, BOTH arms in one process. The window is read LIVE inside
  ``DraftHead._windowed_read_kv`` from ``draft.attn_window_tokens``, so the five
  arms W=0/1024/2048/4096/8192 are run against one engine by mutating that
  attribute in place — no rebuild, no OOM, and on sm70 (eager, no captured graph)
  there is not even a per-shape warm-up to control. W=0 is the full-prefix
  control; every ratio is relative to it.
- SAME prompts for every W arm within a length (paired comparison: a between-W
  difference cannot be a between-passage difference). Prompts are disjoint
  wikitext-103 spans per length (--split, default test). Prefix sharing is off
  (NoPrefixStore), so reusing the same text across arms never hits a cache.
- Direct ``engine.submit(token_ids, ...)`` bypasses the chat template, so the run
  is think-off with no template knob; spec_depth=1.
- COLD FILL is timed separately from decode (submit -> phase DECODE wall), and
  only the post-prefill decode window feeds draft ms / acceptance / tok-s.
- Self-proof: after each step we read ``draft.read_window_stats()``. W=0 must
  never engage; W>0 at 9k+ must engage with a first-page >0 and a windowed
  seq_len that tracks W. The script REFUSES an arm whose self-proof contradicts
  its label, so a sweep over a view that silently went inert (the #668 failure)
  is flagged in the table, not published as flat acceptance.

    python3 -u scripts/probe_draft_window_sweep.py \
        --source $CKPT --draft $CKPT/model_mtp.safetensors \
        --time-draft            # arm the engine's CUDA-event draft seam

--dry-run resolves and prints the plan (arms, lengths, prompts, engine sizing)
without building a model, so the argparse/plan path is exercisable on a CPU box.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

WINDOWS = (0, 1024, 2048, 4096, 8192)
LENGTHS = (9216, 16384, 32768)  # 9k / 16k / 32k prompt context
BLOCK_TOKENS = 16


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sha(path: str) -> str:
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def _engine_sha() -> str:
    for p in (pathlib.Path(".synced_commit"), pathlib.Path("../.synced_commit")):
        if p.exists():
            return p.read_text().strip() or "empty stamp"
    return "no .synced_commit"


def measure_one(eng, draft, prompt_ids, out_tokens):
    """Submit one prompt; return (cold_fill_s, decode stats).

    Cold fill = submit -> the request is in DECODE. The decode window then runs
    exactly ``out_tokens`` (decode_forwards-gated), collecting:
      draft_ms   per-draft-forward CUDA-event ms (only with the timing seam on),
      tick_ms    wall per decode forward (submission-to-submission, no per-tick sync),
      self-proof distinct engaged (sq, first, windowed_seq_len) shapes,
      spec accepted/drafted deltas and generated-token count.
    """
    from tilerl.engine import _PHASE_DECODE, SamplingParams

    rid = eng.submit(
        list(prompt_ids), SamplingParams(temperature=0.0, max_new_tokens=out_tokens, seed=0)
    )
    # ---- cold fill, isolated from the decode window ----
    _sync()
    t_fill0 = time.perf_counter()
    reached = False
    while True:
        eng.step()
        live = [r for r in eng._running if r.req_id == rid]
        if live and all(r.phase == _PHASE_DECODE for r in live):
            reached = True
            break
        if not live and rid in eng.poll():
            break
    _sync()
    cold_fill_s = time.perf_counter() - t_fill0
    if not reached:
        return cold_fill_s, None  # finished in prefill; caller drops it

    # Drop prefill-drain draft timings (the prompt's own forward), mirror of
    # ab_draft_depth.measure.
    if eng._draft_ms is not None:
        eng._draft_ms.clear()

    s0 = eng.stats()
    t0 = time.perf_counter()
    tick_ms, draft_ms, shapes = [], [], set()
    done = {}
    while rid not in done:
        b0, k0 = eng.stats(), time.perf_counter()
        d0 = len(eng._draft_ms or ())
        eng.step()
        st = draft.read_window_stats()
        if st is not None:
            for i in range(len(st["sq"])):
                shapes.add((tuple(st["sq"]), tuple(st["first"]), tuple(st["seq_len"])))
        nf = eng.stats()["decode_forwards"] - b0["decode_forwards"]
        if nf:
            tick_ms.append((time.perf_counter() - k0) * 1000 / nf)
            draft_ms.extend(ms for _, ms, _ in list(eng._draft_ms or ())[d0:])
        done.update({k: v for k, v in eng.poll().items() if k == rid})
    _sync()
    decode_s = time.perf_counter() - t0
    s1 = eng.stats()

    drafted = s1["spec_drafted"] - s0["spec_drafted"]
    accepted = s1["spec_accepted"] - s0["spec_accepted"]
    n_gen = s1["tokens_generated"] - s0["tokens_generated"]
    n_fwd = s1["decode_forwards"] - s0["decode_forwards"]
    return cold_fill_s, {
        "n_gen": n_gen,
        "n_fwd": n_fwd,
        "decode_s": decode_s,
        "tok_s": n_gen / decode_s if decode_s > 0 else 0.0,
        "tick_ms_med": statistics.median(tick_ms) if tick_ms else 0.0,
        "draft_ms_med": statistics.median(draft_ms) if draft_ms else 0.0,
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": accepted / drafted if drafted else 0.0,
        "accept_len": n_gen / n_fwd if n_fwd else 0.0,
        "shapes": sorted(shapes),
    }


def aggregate(w, cold, dec, expect_engage):
    """Median across prompts for one (length, W) arm, plus the self-proof verdict."""
    if not dec:
        return None
    shapes = set()
    for d in dec:
        shapes.update(d["shapes"])
    engaged = any(any(f > 0 for f in first) for _, first, _ in shapes)
    # W=0 must NOT engage; W>0 at these lengths MUST truncate. A mismatch is the
    # exact #668 inert-view failure and is surfaced, not hidden.
    if expect_engage and not engaged:
        proof = "FAIL:window-never-truncated"
    elif not expect_engage and shapes:
        proof = "FAIL:W0-engaged"
    else:
        proof = "ok"
    return {
        "W": w,
        "cold_fill_s_med": statistics.median(cold),
        "draft_ms_med": statistics.median(d["draft_ms_med"] for d in dec),
        "accept_rate": statistics.mean(d["accept_rate"] for d in dec),
        "accept_len": statistics.mean(d["accept_len"] for d in dec),
        "tok_s_med": statistics.median(d["tok_s"] for d in dec),
        "proof": proof,
        "n_prompts": len(dec),
        "shapes_sample": [list(map(list, s)) for s in sorted(shapes)[:3]],
    }


def _write_json(path, table) -> None:
    """Atomic, incremental table write (tmp + replace): a crash in a LATER length
    cannot destroy the lengths already finished. Called after every length."""
    if not path:
        return
    p = pathlib.Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(table, indent=2))
    os.replace(tmp, p)


def plan_corpus(stream, lengths, want):
    """Per-length ``(spans, n_eff)`` from one token stream, adapting n to corpus
    size so a long ctx with too few disjoint spans does not crash. Disjoint
    independent prompts (n_eff = min(want, (corpus-skip)//ctx)). The same span
    list per length is reused across every W arm, so arms stay paired."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import tiled_spans  # noqa: E402

    # skip=512 drops wikitext's header/newline head (corpus convention). Across
    # lengths prompts need not be disjoint (separate analyses; prefix sharing is
    # off), and skipping a full `length` at 32k would waste a scarce window.
    return {length: tiled_spans(stream, want, length, skip=512) for length in lengths}


def run(args) -> list[dict]:
    from tilerl_kernels.backend import get_backend

    import tilerl.build as build
    from tilerl.build import NoPrefixStore, build_engine, build_model
    from tilerl.spec import load_draft
    from tilerl.tokenizer import get_tokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from corpus import wikitext_ids_stream  # noqa: E402

    os.environ.setdefault("TILERL_TARGET", "cuda")
    os.environ.setdefault("TILERL_QWEN38_SOURCE", args.source)
    build.QWEN38_SOURCE = args.source

    # Sample-size gate BEFORE building the 27B: the corpus read needs only the
    # tokenizer, so a half-present split fails fast instead of after a long load.
    tok = get_tokenizer(args.source)
    stream = wikitext_ids_stream(tok, args.split, args.corpus_glob)
    corpus_plan = plan_corpus(stream, args.lengths, args.prompts)
    del stream
    if args.min_prompts_per_length > 0:
        short = {
            length: n_eff
            for length, (_, n_eff) in corpus_plan.items()
            if n_eff < args.min_prompts_per_length
        }
        if short:
            got = ", ".join(f"ctx={L}: {n}/{args.min_prompts_per_length}" for L, n in short.items())
            raise SystemExit(
                f"corpus split {args.split!r} yields too few disjoint prompts: {got}; "
                "refusing to publish a W confirmation under --min-prompts-per-length"
            )

    be = get_backend()
    cfg, model = build_model("qwen38-27b", seed=0, fuse_projections=True)
    draft = load_draft(model, args.draft)

    max_len = max(args.lengths)
    need_blocks = -(-(max_len + args.out_tokens) // BLOCK_TOKENS) + 8
    eng = build_engine(
        cfg,
        model,
        be,
        num_blocks=need_blocks,
        num_slots=2,
        max_batch=1,
        max_total_tokens=max_len + args.out_tokens + 64,
        draft=draft,
        spec_depth=1,
        sparse_k=args.sparse_k,
        prefix_store=NoPrefixStore(),
    )
    if args.time_draft:
        # Direct assignment is what arms the seam on an already-built engine; the
        # env is read only in Engine.__init__ (too late to set here).
        eng._draft_ms = []

    arch = getattr(be, "arch", "") or "sm70"
    print(f"# probe {_sha(__file__)}, engine tree {_engine_sha()}, arch {arch}")
    print(
        f"# one engine; W mutated in place; split={args.split}, lengths={args.lengths}, "
        f"windows={args.windows}, requested prompts/len={args.prompts}, "
        f"out_tokens={args.out_tokens}, sparse_k={args.sparse_k}, prefix=off"
    )

    table = []
    for length in args.lengths:
        prompts, n_eff = corpus_plan[length]
        print(
            f"# context={length}: n_eff={n_eff}/{args.prompts} independent prompts"
            + (
                "  (corpus-limited; median still valid, sample size labelled)"
                if n_eff < args.prompts
                else ""
            )
        )
        if not prompts:
            print(f"# context={length}: corpus too small even for one span; skipping")
            continue
        try:
            per_arm = {}
            for w in args.windows:
                draft.attn_window_tokens = w
                cold, dec = [], []
                for p in prompts:
                    cf, d = measure_one(eng, draft, p, args.out_tokens)
                    cold.append(cf)
                    if d is not None:
                        dec.append(d)
                row = aggregate(w, cold, dec, expect_engage=w > 0)
                row["length"] = length
                row["n_requested"] = args.prompts
                per_arm[w] = row
                table.append(row)
            # ratios vs the W=0 control for this length
            base = per_arm.get(0) or {}
            for row in (per_arm[w] for w in args.windows):
                bt = base.get("tok_s_med") or 0.0
                bd = base.get("draft_ms_med") or 0.0
                row["tok_s_ratio_vs_W0"] = row["tok_s_med"] / bt if bt else 0.0
                row["draft_ms_ratio_vs_W0"] = row["draft_ms_med"] / bd if bd else 0.0
            _print_length(length, per_arm, args.windows)
        except Exception as exc:  # one length failing must keep earlier lengths' JSON
            print(
                f"# context={length}: ARM FAILED, keeping prior lengths: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            table.append({"length": length, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            _write_json(args.json, table)  # incremental: flush after EVERY length
    draft.attn_window_tokens = 0
    if args.json:
        print(f"# wrote {args.json} ({len(table)} rows)")
    return table


def _print_length(length, per_arm, windows):
    print(f"\n# context={length}  (ratios relative to W=0)")
    print(
        f"# {'W':>5} {'cold_s':>8} {'draft_ms':>9} {'draft_x':>8} "
        f"{'accept':>7} {'acc_len':>8} {'tok/s':>8} {'tok/s_x':>8}  proof"
    )
    for w in windows:
        r = per_arm.get(w)
        if r is None:
            print(f"{w:>5}  (no decode rows)")
            continue
        print(
            f"{w:>5} {r['cold_fill_s_med']:8.3f} {r['draft_ms_med']:9.3f} "
            f"{r['draft_ms_ratio_vs_W0']:8.3f} {r['accept_rate']:7.3f} "
            f"{r['accept_len']:8.3f} {r['tok_s_med']:8.2f} "
            f"{r['tok_s_ratio_vs_W0']:8.3f}  {r['proof']}  "
            f"n={r['n_prompts']} {r['shapes_sample']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="checkpoint dir (27B); required unless --dry-run")
    ap.add_argument("--draft", help="model_mtp.safetensors path; required unless --dry-run")
    ap.add_argument(
        "--lengths",
        default=",".join(str(x) for x in LENGTHS),
        help="comma-separated prompt contexts",
    )
    ap.add_argument(
        "--windows",
        default=",".join(str(x) for x in WINDOWS),
        help="comma-separated W tokens; 0 = full-prefix control",
    )
    ap.add_argument(
        "--prompts",
        type=int,
        default=30,
        help="unique wikitext spans per length, reused across W arms (paired)",
    )
    ap.add_argument("--out-tokens", type=int, default=96)
    ap.add_argument(
        "--sparse-k",
        type=int,
        default=0,
        help="trunk sparse_k; pass the serve value to mirror the V100 line",
    )
    ap.add_argument(
        "--time-draft",
        action="store_true",
        help="arm the CUDA-event draft_step seam (draft_ms columns)",
    )
    ap.add_argument(
        "--json",
        default="",
        help="optional JSON table path; written "
        "incrementally (tmp+replace) once per length, so a later-length "
        "crash keeps earlier lengths; always a full JSON array (overwritten "
        "atomically, not appended)",
    )
    ap.add_argument(
        "--dry-run-corpus-tokens",
        type=int,
        default=297054,
        help="corpus token count used only to preview n_eff in --dry-run "
        "(wikitext-103 TEST split measured on V100); ignored for real runs",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=("test", "train", "validation"),
        help="wikitext-103 split (corpus.py); test default is unchanged. Train is "
        "large enough for n>=30 disjoint 32k spans; test yields only 9.",
    )
    ap.add_argument(
        "--corpus-glob",
        default="",
        help="expanded parquet glob fully overriding --split (corpus.py)",
    )
    ap.add_argument(
        "--min-prompts-per-length",
        type=int,
        default=30,
        help="hard gate: fail nonzero before the model build if any scanned length "
        "has fewer than this many disjoint prompts, naming the real n_eff (0 = off).",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="resolve/validate the plan and exit; no model build"
    )
    args = ap.parse_args()
    args.lengths = tuple(int(x) for x in args.lengths.split(","))
    args.windows = tuple(int(x) for x in args.windows.split(","))

    if args.prompts < 1:
        raise SystemExit("--prompts must be >= 1")
    if args.out_tokens < 32 or args.out_tokens > 128:
        raise SystemExit(f"--out-tokens should be in [32,128], got {args.out_tokens}")
    if 0 not in args.windows:
        raise SystemExit("the W=0 full-prefix control arm is required")
    if any(w < 0 for w in args.windows):
        raise SystemExit("windows must be >= 0")

    max_len = max(args.lengths)
    need_blocks = -(-(max_len + args.out_tokens) // BLOCK_TOKENS) + 8
    print(f"[dry-run] probe {_sha(__file__) if __file__ else 'n/a'}")
    print(
        f"[dry-run] split={args.split} lengths={args.lengths} windows={args.windows} "
        f"requested prompts/len={args.prompts} out_tokens={args.out_tokens} "
        f"sparse_k={args.sparse_k} time_draft={args.time_draft} "
        f"corpus_tokens={args.dry_run_corpus_tokens if args.split == 'test' else '<resolved at run>'}"
    )
    print(f"[dry-run] engine num_blocks~{need_blocks} (sized for {max_len}+{args.out_tokens})")
    if args.min_prompts_per_length > 0:
        print(
            f"[dry-run] hard gate --min-prompts-per-length={args.min_prompts_per_length}: "
            "a real run fails nonzero if any length has fewer disjoint prompts "
            "(test split yields only 9 at 32k -> pass --min-prompts-per-length 0 for a test run)"
        )
    if args.split != "test":
        # Train/validation token count is not hardcoded; n_eff is whatever the
        # on-box parquet yields. Only bound it by --prompts here.
        print(
            f"[dry-run] split={args.split}: n_eff resolved at run from the on-box "
            f"parquet (target {args.prompts}/length); verify the {args.split} "
            "parquet is in the HF cache before the window"
        )
    else:
        # Preview adaptive n_eff with a synthetic stream of the known test corpus
        # size (no parquet/model). n_eff=min(prompts,(corpus-512)//ctx): 30/18/9.
        preview = plan_corpus(list(range(args.dry_run_corpus_tokens)), args.lengths, args.prompts)
        total_runs = 0
        for length in args.lengths:
            _, n_eff = preview[length]
            total_runs += n_eff
            limited = "  [corpus-limited]" if n_eff < args.prompts else ""
            print(
                f"[dry-run]   ctx={length}: n_eff={n_eff}/{args.prompts} independent "
                f"disjoint prompts{limited}"
            )
        print(
            f"[dry-run] total submit runs={total_runs} x {len(args.windows)} W-arms "
            f"= {total_runs * len(args.windows)} measurements"
        )
    print(
        "[dry-run] paired disjoint prompts across W arms; NoPrefixStore; direct "
        "submit=think-off, spec_depth=1; cold fill timed separately; JSON flushed "
        "atomically after every length (full array, post-processing reads it directly)"
    )
    print(
        "[dry-run] TODO(if requested): a W-only vs +page0-anchor arm needs a "
        "page0-attn switch the engine does not currently expose; add one before "
        "that comparison, do not fake it here."
    )
    if args.dry_run:
        return
    if not args.source or not args.draft:
        raise SystemExit("--source and --draft are required for a real run")
    run(args)


if __name__ == "__main__":
    main()
