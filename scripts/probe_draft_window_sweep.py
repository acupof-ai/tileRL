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
- PER-PROMPT SPREAD, because a cross-prompt median hides the variance a verdict
  rests on. Each prompt's own decode ticks are kept individually, so the table
  carries per-prompt p10/p50/p90/IQR and a cross-prompt distribution of the
  per-prompt tok/s, not one median over prompts. The tick set is the serve-line
  steady set (``dec=1 & sparse=1 & model>0 & sample>0``): idle/graph ticks and
  the request's own closing sample tick are excluded, the latter single-listed
  with its model/sample split so a misclassification is visible rather than
  silent. ``TILERL_STEP_TIMING`` is armed in-process to read those segments.
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
--self-check runs the hermetic spread/steady-set check on synthetic ticks (no
card); tests/test_wsweep_prompt_spread.py calls it and controls it.
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


# --- per-tick reading --------------------------------------------------------
#
# The steady set is the serve line's own reading discipline: dec=1 & sparse=1 &
# model>0 & sample>0, path != graph. It drops idle/untimed ticks AND a request's
# closing tick, whose sample segment takes over the whole tick; that one is
# single-listed rather than dropped, because dropping it silently would flatter
# the steady band it is being compared against.


def _pct(xs, q: float) -> float:
    """Nearest-rank percentile, the convention probe_headroom_coldtail reads the
    same serve logs with, so the two instruments agree on a quantile."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]


def spread(xs) -> dict:
    """p10/p50/p90 + IQR over one sample. IQR is p75-p25 on the same
    nearest-rank convention, so a 4-tick prompt still has a defined band
    instead of a null."""
    return {
        "n": len(xs),
        "p10": _pct(xs, 0.10),
        "p50": _pct(xs, 0.50),
        "p90": _pct(xs, 0.90),
        "iqr": _pct(xs, 0.75) - _pct(xs, 0.25),
    }


def classify_tick(tm, want_sparse: bool) -> dict | None:
    """One decode tick's segments, or None when it is not a steady serve tick.

    ``tm`` is the engine's ``_StepTiming`` after ``step()`` returned: ``cur`` is
    cleared in ``tick_start`` and re-read in ``tick_end``, so it still describes
    the tick that just ran, with no engine hook.
    """
    if tm is None or not tm.phase_dec:
        return None
    if tm.fwd_path == "graph" or tm.fwd_sparse != want_sparse:
        return None
    model_ms = tm.cur.get("model", 0.0) * 1000.0
    sample_ms = tm.cur.get("sample", 0.0) * 1000.0
    if model_ms <= 0 or sample_ms <= 0:
        return None
    total_ms = tm.last_total * 1000.0
    return {
        "total_ms": total_ms,
        "model_ms": model_ms,
        "sample_ms": sample_ms,
        "closing": sample_ms > 0.5 * total_ms,
    }


def _sha(path: str) -> str:
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def _engine_sha() -> str:
    for p in (pathlib.Path(".synced_commit"), pathlib.Path("../.synced_commit")):
        if p.exists():
            return p.read_text().strip() or "empty stamp"
    return "no .synced_commit"


def measure_one(eng, draft, prompt_ids, out_tokens, want_sparse):
    """Submit one prompt; return (cold_fill_s, decode stats).

    Cold fill = submit -> the request is in DECODE. The decode window then runs
    exactly ``out_tokens`` (decode_forwards-gated), collecting:
      draft_ms   per-draft-forward CUDA-event ms (only with the timing seam on),
      tick_ms    wall per decode forward (submission-to-submission, no per-tick sync),
      self-proof distinct engaged (sq, first, windowed_seq_len) shapes,
      spec accepted/drafted deltas and generated-token count,
      per-tick segments for the steady-set spread (``want_sparse`` is the arm's
      own ``--sparse-k > 0``, so the filter asserts the engine ran what the arm
      asked it to run rather than assuming it).
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
    steady_ms, closing = [], []
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
        tk = classify_tick(eng._step_timing, want_sparse=want_sparse)
        if nf:
            tick_ms.append((time.perf_counter() - k0) * 1000 / nf)
            draft_ms.extend(ms for _, ms, _ in list(eng._draft_ms or ())[d0:])
            if tk is not None:
                (closing if tk["closing"] else steady_ms).append(tk)
        done.update({k: v for k, v in eng.poll().items() if k == rid})
    _sync()
    decode_s = time.perf_counter() - t0
    s1 = eng.stats()

    drafted = s1["spec_drafted"] - s0["spec_drafted"]
    accepted = s1["spec_accepted"] - s0["spec_accepted"]
    n_gen = s1["tokens_generated"] - s0["tokens_generated"]
    n_fwd = s1["decode_forwards"] - s0["decode_forwards"]
    # Per-prompt steady tick set -> this prompt's own tok/s band. tok/s is
    # tokens-per-forward over the steady forward's own wall, so it is per-prompt
    # by construction and does not inherit the whole-request decode span (which
    # contains the closing tick the steady set excludes).
    steady_tot = [t["total_ms"] for t in steady_ms]
    per_fwd_tok = n_gen / n_fwd if n_fwd else 0.0
    prompt_tok_s = [per_fwd_tok * 1000.0 / ms for ms in steady_tot if ms > 0]
    return cold_fill_s, {
        "n_gen": n_gen,
        "n_fwd": n_fwd,
        "prompt_len": len(prompt_ids),
        "decode_s": decode_s,
        "tok_s": n_gen / decode_s if decode_s > 0 else 0.0,
        "tick_ms_med": statistics.median(tick_ms) if tick_ms else 0.0,
        "draft_ms_med": statistics.median(draft_ms) if draft_ms else 0.0,
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": accepted / drafted if drafted else 0.0,
        "accept_len": n_gen / n_fwd if n_fwd else 0.0,
        "shapes": sorted(shapes),
        "steady_ticks": len(steady_ms),
        "closing_ticks": len(closing),
        "tick_spread": spread(steady_tot),
        "tok_s_spread": spread(prompt_tok_s),
        # Closing ticks are reported, never folded into the band above. A
        # classifying miss here (sample not dominant) would otherwise be
        # invisible; the model/sample pair makes it readable.
        "closing_sample": [
            {"total_ms": round(t["total_ms"], 1), "model_ms": round(t["model_ms"], 1),
             "sample_ms": round(t["sample_ms"], 1)}
            for t in closing[:3]
        ],
    }


def aggregate(w, cold, dec, expect_engage):
    """Cross-prompt summary for one (length, W) arm, plus the self-proof verdict.

    Reports the per-prompt tok/s distribution, not just a median over prompts:
    each prompt contributes its own steady-band median, and the arm's spread is
    taken over those per-prompt values. A prompt whose steady set is empty is
    counted and named rather than averaged in as a zero.
    """
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
    per_prompt = [d["tok_s_spread"]["p50"] for d in dec if d["steady_ticks"]]
    no_steady = len(dec) - len(per_prompt)
    band = spread(per_prompt)
    # The per-prompt rows themselves, not only their summary: a verdict that
    # rests on the within-arm distribution has to be able to READ that
    # distribution, and a later run has to be comparable prompt by prompt.
    per_prompt_rows = [
        {
            "i": i,
            "prompt_len": d.get("prompt_len", 0),
            "steady_ticks": d["steady_ticks"],
            "closing_ticks": d["closing_ticks"],
            "tok_s": d["tok_s_spread"],
            "tick_ms": d["tick_spread"],
            "accept_rate": d["accept_rate"],
            "accept_len": d["accept_len"],
        }
        for i, d in enumerate(dec)
    ]
    return {
        "W": w,
        "cold_fill_s_med": statistics.median(cold),
        "draft_ms_med": statistics.median(d["draft_ms_med"] for d in dec),
        "accept_rate": statistics.mean(d["accept_rate"] for d in dec),
        "accept_len": statistics.mean(d["accept_len"] for d in dec),
        "tok_s_med": statistics.median(d["tok_s"] for d in dec),
        # The headline the verdict needs: how far apart the PROMPTS are, not how
        # far apart the ticks within one prompt are. `prompt_tok_s` is the
        # summary; `per_prompt` below is the evidence behind it.
        "prompt_tok_s": band,
        "prompt_tok_s_min": min(per_prompt) if per_prompt else 0.0,
        "prompt_tok_s_max": max(per_prompt) if per_prompt else 0.0,
        "prompt_tok_s_iqr_frac": band["iqr"] / band["p50"] if band["p50"] else 0.0,
        "per_prompt": per_prompt_rows,
        "steady_ticks_total": sum(d["steady_ticks"] for d in dec),
        "steady_ticks_per_prompt_med": (
            statistics.median([d["steady_ticks"] for d in dec if d["steady_ticks"]])
            if per_prompt else 0
        ),
        "closing_ticks_total": sum(d["closing_ticks"] for d in dec),
        "prompts_without_steady_ticks": no_steady,
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


def self_check() -> int:
    """Hermetic: the spread/quantile logic and the steady-set filter, on
    synthetic ticks. Imported and called by tests/test_wsweep_prompt_spread.py;
    the module is not in the CI hermetic set (it names a backend), so this is the
    only thing that runs it without a card."""
    import types

    def tm(dec=1, path="eager", sparse=True, model=0.165, sample=0.001, total=0.180):
        return types.SimpleNamespace(
            phase_dec=dec, fwd_path=path, fwd_sparse=sparse,
            cur={"model": model, "sample": sample}, last_total=total,
        )

    # Quantile convention: nearest-rank, so a 4-sample prompt still has a band.
    assert _pct([100, 200, 300, 400], 0.5) == 200
    assert _pct([100, 200, 300, 400], 0.25) == 100
    assert _pct([100, 200, 300, 400], 0.75) == 300
    assert _pct([], 0.5) == 0.0
    s = spread([100, 200, 300, 400])
    # nearest-rank on 4 samples: p90 lands on index min(3, int(0.9*3)) = 2.
    assert s == {"n": 4, "p10": 100, "p50": 200, "p90": 300, "iqr": 200}, s
    # One sample is a degenerate band, not a crash and not an invented spread.
    assert spread([250]) == {"n": 1, "p10": 250, "p50": 250, "p90": 250, "iqr": 0}

    # The steady set: exactly the serve-line filter (dec=1 & sparse=1 &
    # model>0 & sample>0, path != graph).
    assert classify_tick(None, want_sparse=True) is None
    assert classify_tick(tm(dec=0), want_sparse=True) is None          # prefill/idle
    assert classify_tick(tm(path="graph"), want_sparse=True) is None   # captured tick
    assert classify_tick(tm(sparse=False), want_sparse=True) is None   # dense arm tick
    assert classify_tick(tm(sparse=True), want_sparse=False) is None
    assert classify_tick(tm(sample=0.0), want_sparse=True) is None     # idle: no sample
    assert classify_tick(tm(model=0.0), want_sparse=True) is None      # idle: no model
    ok = classify_tick(tm(), want_sparse=True)
    assert ok is not None and not ok["closing"]
    # A request's closing tick: model steady, sample owns the tick. It must be
    # flagged, because folding it into the band would drag the p90 down.
    closing = classify_tick(tm(model=0.165, sample=5.440, total=5.700), want_sparse=True)
    assert closing is not None and closing["closing"]

    # The statistic the verdict needs is the PER-PROMPT median, and the toy data
    # below is built so a mean-based substitute collapses to flat while the
    # median does not. Two prompts are tight (ticks all 100 ms); two have three
    # 10 ms ticks plus one 370 ms outlier: their MEAN tick is also 100 ms but
    # their MEDIAN tick stays 10 ms (median tok/s 160 vs the tight prompts' 16).
    # A per-prompt-median band separates the two groups; anything computed off
    # the mean reports all four prompts as identical. Nearest-rank IQR needs four
    # samples, hence four prompts.
    def dec_row(ticks_ms, n_gen=32, n_fwd=20):
        per_fwd_tok = n_gen / n_fwd
        return {
            "draft_ms_med": 11.0, "accept_rate": 0.73, "accept_len": 1.72,
            "tok_s": per_fwd_tok * 1000 / (sum(ticks_ms) / len(ticks_ms)),
            "shapes": [(tuple([8]), tuple([2048]), tuple([2048]))],
            "steady_ticks": len(ticks_ms), "closing_ticks": 1,
            "tick_spread": spread(ticks_ms),
            "tok_s_spread": spread([per_fwd_tok * 1000 / m for m in ticks_ms]),
        }

    tight = [100, 100, 100, 100]      # median 100 ms, mean 100 ms
    spiky = [10, 10, 10, 370]         # median 10 ms,  mean 100 ms
    same_mean = aggregate(
        2048, [3.0], [dec_row(tight), dec_row(tight), dec_row(spiky), dec_row(spiky)],
        expect_engage=True,
    )
    all_tight = aggregate(2048, [3.0], [dec_row(tight)] * 4, expect_engage=True)

    # Four prompts, identical MEAN tick in both arms -> the mean-based view can
    # never separate them, which is what makes the assertion below load-bearing.
    means = {round(dec_row(t)["tok_s"], 9) for t in (tight, spiky)}
    assert len(means) == 1, means
    assert all_tight["prompt_tok_s"]["iqr"] == 0.0, all_tight["prompt_tok_s"]
    assert same_mean["prompt_tok_s"]["iqr"] > 0.0, same_mean["prompt_tok_s"]
    assert same_mean["prompt_tok_s_max"] > same_mean["prompt_tok_s_min"]
    assert same_mean["prompt_tok_s_iqr_frac"] > 0.1, same_mean["prompt_tok_s_iqr_frac"]
    # Within a prompt the ticks are tight in the tight prompts: the separation
    # above is BETWEEN prompts, not a within-prompt tail bleeding through.
    assert spread(tight)["iqr"] == 0.0
    # A prompt with no steady tick is counted, never averaged in as a zero.
    none_steady = dec_row(tight)
    none_steady["steady_ticks"] = 0
    none_steady["tok_s_spread"] = spread([])
    one_short = aggregate(2048, [3.0], [dec_row(tight), none_steady], expect_engage=True)
    assert one_short["prompts_without_steady_ticks"] == 1, one_short
    assert one_short["prompt_tok_s_min"] > 0
    # The rendered table is part of the deliverable, so a key the printer reads
    # but the row dict does not define must fail here, not on the card after a
    # multi-hour sweep. Captured, because the printers write to stdout.
    import contextlib
    import io

    for r in (same_mean, all_tight, one_short):
        r["draft_ms_ratio_vs_W0"] = 1.0
        r["tok_s_ratio_vs_W0"] = 1.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_length(32768, {2048: same_mean, 0: one_short}, (0, 2048))
    out = buf.getvalue()
    assert "per-prompt rows" in out and "prompt_len" in out, out
    # Every per-prompt row is rendered, and the no-steady prompt in `one_short`
    # renders as '-' on the band columns rather than as a zero.
    assert out.count(" 2048 ") >= 4, out
    assert out.count("      - ") >= 4, out
    print("probe_draft_window_sweep self-check ok")
    return 0


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
    # The steady-set filter reads the engine's per-tick segments, which exist
    # only when the timing seam is on. Armed before the build (Engine.__init__
    # reads the env); SLOW_MS=0 so EVERY tick is recorded, not just the slow
    # tail -- a >threshold-only log is not a distribution.
    os.environ["TILERL_STEP_TIMING"] = "1"
    os.environ["TILERL_STEP_TIMING_SLOW_MS"] = "0"

    # Sample-size gate BEFORE building the 27B: the corpus read needs only the
    # tokenizer, so a half-present split fails fast instead of after a long load.
    tok = get_tokenizer(args.source)
    # Bound tokenization to what the largest length's n disjoint spans consume
    # (skip=512 + n*ctx) plus one page margin. Without it the full train split
    # (~540M chars) is one tok.encode -> ~140M Python ints -> host OOM (rc137).
    need_tokens = 512 + args.prompts * max(args.lengths) + 4096
    stream = wikitext_ids_stream(tok, args.split, args.corpus_glob, need_tokens)
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
                    cf, d = measure_one(eng, draft, p, args.out_tokens, args.sparse_k > 0)
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
    # Per-prompt spread: the arm's tok/s band ACROSS PROMPTS, which the median
    # above cannot show. An arm whose prompts disagree by more than the between-W
    # gap being decided on is not a decision this table can carry.
    print(
        f"# {'W':>5} {'prompt tok/s':>28}  {'p10':>7} {'p50':>7} {'p90':>7} "
        f"{'IQR/p50':>8} {'min':>7} {'max':>7}"
    )
    for w in windows:
        r = per_arm.get(w)
        if r is None:
            continue
        s = r["prompt_tok_s"]
        print(
            f"{w:>5} {'per-prompt band':>28}  {s['p10']:7.2f} {s['p50']:7.2f} "
            f"{s['p90']:7.2f} {r['prompt_tok_s_iqr_frac']:8.3f} "
            f"{r['prompt_tok_s_min']:7.2f} {r['prompt_tok_s_max']:7.2f}"
            f"   ticks steady={r['steady_ticks_total']}"
            f" closing={r['closing_ticks_total']}"
            + (
                f"  NO-STEADY-TICKS on {r['prompts_without_steady_ticks']} prompt(s)"
                if r["prompts_without_steady_ticks"]
                else ""
            )
        )
    # The evidence behind the band above, one row per prompt: a verdict that
    # rests on the within-arm distribution must be readable prompt by prompt.
    print(f"# per-prompt rows (ctx={length}); '-' = no steady tick for that prompt")
    print(
        f"# {'W':>5} {'i':>3} {'prompt_len':>10} {'steady':>7} {'close':>6} "
        f"{'p10':>7} {'p50':>7} {'p90':>7} {'IQR':>7} {'accept':>7} {'acc_len':>8}"
    )
    for w in windows:
        r = per_arm.get(w)
        if r is None:
            continue
        for row in r["per_prompt"]:
            s, t = row["tok_s"], row["tick_ms"]
            print(
                f"{w:>5} {row['i']:>3} {row['prompt_len']:>10} "
                f"{row['steady_ticks']:>7} {row['closing_ticks']:>6} "
                + (
                    f"{s['p10']:7.2f} {s['p50']:7.2f} {s['p90']:7.2f} "
                    f"{s['iqr']:7.2f} {row['accept_rate']:7.3f} "
                    f"{row['accept_len']:8.3f}"
                    if row["steady_ticks"]
                    else f"{'-':>7} {'-':>7} {'-':>7} {'-':>7} "
                         f"{row['accept_rate']:7.3f} {row['accept_len']:8.3f}"
                )
                + (f"   tick p50={t['p50']:.0f}ms" if row["steady_ticks"] else "")
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
        "--self-check",
        action="store_true",
        help="run the hermetic spread/steady-set self-check and exit (no card)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="resolve/validate the plan and exit; no model build"
    )
    args = ap.parse_args()
    if args.self_check:
        return self_check()
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
        "[dry-run] TILERL_STEP_TIMING=1 SLOW_MS=0 armed in-process: every tick is "
        f"recorded so the steady set (dec=1 & sparse={int(bool(args.sparse_k))} & "
        "model>0 & sample>0, path != graph) is a distribution, not a slow tail; "
        "the closing tick is counted separately, never folded into the band."
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
    raise SystemExit(main())
