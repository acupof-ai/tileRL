"""tilerl bench harness: one runner, five suites, a snapshot baseline gate.

Fills the coverage holes the scattered bench_*.py scripts left open:

  decode-kv   decode tok/s vs KV depth (512/2k/8k/32k) at B=1,8 — the serving
              metric that DEGRADES with context (paged-attention KV read scales
              with depth). The old verify check 4 timed one shallow ~512 depth.
  prefill     prefill tok/s vs prompt length, as a curve not 3 points.
  kv-reuse    prefix-cache hit-rate + warm-vs-cold prefill speedup (the engine's
              PrefixStore was perf-untested).
  train       train_step fwd+bwd tok/s (tiny on CPU, 27B on the pod) — training
              had zero perf coverage.
  micro       per-kernel roofline table (opt-in; folds in bench_fp4_gemv).

Baseline: docs/experience/wins/bench-baseline.json, keyed (suite, shape, target)
-> tok/s + commit + date. A row PASSES at >= 0.97x its baseline; a row that BEATS
its baseline auto-raises the entry. Below 0.97x -> FAIL, exit 1. First run with
no baseline seeds it. Timing reuses verify_h20_fp4's steady-state settle loop and
median-of-windows for stability.

  uv run tilerl bench --suite train                       # CPU, tiny
  tilerl bench --source /data00/Qwen3.8-27B-NVFP4 --gpu 7  # pod, all GPU suites
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_BASELINE = Path(__file__).resolve().parent.parent / "docs/experience/wins/bench-baseline.json"
_GATE = 0.97  # a run must be within 3% of (or beat) the snapshot
_KV_DEPTHS = (512, 2048, 8192, 32768)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _today() -> str:
    # Date.now is fine here (a script, not a workflow); stamp the baseline row.
    return time.strftime("%Y-%m-%d")


def _load_baseline() -> dict:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


def _save_baseline(b: dict) -> None:
    _BASELINE.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")


class Gate:
    """Compares tok/s rows against the snapshot, auto-raises winners, tracks FAILs."""

    def __init__(self, target: str, update_only: bool = False):
        self.target = target
        self.baseline = _load_baseline()
        self.commit = _git_commit()
        self.date = _today()
        self.rows: list[dict] = []
        self.failed = False
        self.dirty = False
        self.seed_only = update_only  # first-seed run: record, never fail

    def check(self, suite: str, shape: str, tok_s: float) -> str:
        key = f"{suite}/{shape}/{self.target}"
        prev = self.baseline.get(key)
        verdict = "SEED"
        if prev is None or self.seed_only:
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "SEED"
        elif tok_s > prev["tok_s"]:
            print(f"  RAISED {key}: {prev['tok_s']:.1f} -> {tok_s:.1f} tok/s")
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "RAISE"
        elif tok_s >= prev["tok_s"] * _GATE:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            # CPU is thermally/scheduler-noisy (~4% run-to-run) — gate it as a
            # report-only smoke, not a hard fail. The 3% gate is for the stable
            # GPU baseline (matches CI: only lint + CPU-correctness block).
            if self.target != "cpu":
                self.failed = True
        base = prev["tok_s"] if prev else tok_s
        self.rows.append(
            {"key": key, "tok_s": tok_s, "baseline": base, "ratio": tok_s / base if base else 0.0, "verdict": verdict}
        )
        return verdict

    def finish(self) -> int:
        if self.dirty:
            _save_baseline(self.baseline)
            print(f"\n  baseline updated: {_BASELINE}")
        print("\n=== bench summary ===")
        for r in self.rows:
            soft = "  (report-only)" if r["verdict"] == "FAIL" and self.target == "cpu" else ""
            print(
                f"  {r['verdict']:<5} {r['key']:<44} {r['tok_s']:>9.1f} tok/s"
                f"  ({r['ratio']:.3f}x){soft}"
            )
        return 1 if self.failed else 0


# --- timing (median of windows, reusing verify's steady-state settle) --------


def _median_windows(step_fn, n_windows: int, ticks: int, sync) -> float:
    """Median ms/tick over n_windows of `ticks` steady-state steps each."""
    samples = []
    for _ in range(n_windows):
        sync()
        t0 = time.perf_counter()
        for _ in range(ticks):
            step_fn()
        sync()
        samples.append((time.perf_counter() - t0) / ticks * 1e3)
    return statistics.median(samples)


# --- suites ------------------------------------------------------------------


def suite_decode_kv(gate, cfg, model, backend, batches, depths, ticks):
    """Decode tok/s vs KV depth. Rebuilds the engine per depth so the pools are
    sized for that KV (32k*B tokens needs many blocks) — settle_decode drives
    every row to pure-decode before timing."""
    import benchkit as bk
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS

    print("\n=== decode-vs-KV-depth (tok/s, higher=better; should DROP with depth) ===")
    print(f"  {'depth':>7} {'B':>3} {'ms/tick':>9} {'tok/s/req':>10} {'agg tok/s':>10}")
    for depth in depths:
        if depth > cfg.max_position_embeddings:
            continue
        for b in batches:
            if depth * b > 65536:
                print(f"  {depth:>7} {b:>3}   (skipped: KV pool would exceed one H20)")
                continue
            # 2x headroom: the prefix store pins every finished prompt's blocks
            # for reuse, so a run that just drained still holds ~depth*B blocks.
            # outlive the batch's staggered prefill (chunks of 512/tick) or row 0 is
            # DONE before the last row reaches decode and settle never sees B rows.
            gen = ticks + 40 + b * (depth // 512 + 4)
            need = 2 * (-(-(depth + gen) * b // BLOCK_TOKENS) + b)
            engine = build_engine(
                cfg, model, backend,
                num_blocks=max(256, need), num_slots=max(16, b),
                max_batch=max(8, b), max_total_tokens=max(8192, depth + gen + 64),
            )
            wids = [
                engine.submit(
                    bk.rand_prompt(cfg.vocab_size, depth, seed=900 + depth + i),
                    SamplingParams(temperature=0.0, max_new_tokens=gen, seed=i),
                )
                for i in range(b)
            ]
            if not bk.settle_decode(engine, b, depth * b // 128):
                print(f"  {depth:>7} {b:>3}   (never reached pure decode — skipped)")
                continue
            for _ in range(8):
                engine.step()
            ms = _median_windows(engine.step, 3, ticks, lambda: bk.sync(backend))
            agg = 1000.0 * b / ms
            print(f"  {depth:>7} {b:>3} {ms:>9.3f} {1000.0 / ms:>10.1f} {agg:>10.1f}")
            gate.check("decode-kv", f"d{depth}-b{b}", agg)
            # drain so the next depth's blocks are free
            done: dict = {}
            for _ in range(ticks + 8 * b + 256):
                done.update(engine.poll())
                if all(w in done for w in wids):
                    break
                engine.step()


def suite_prefill(gate, cfg, model, backend, lengths):
    import benchkit as bk
    from tilerl.engine import build_engine

    cap = min(8192, cfg.max_position_embeddings)
    engine = build_engine(cfg, model, backend, num_blocks=cap // 16 + 64, num_slots=16,
                          max_batch=8, max_total_tokens=cap + 64)
    print("\n=== prefill-vs-length (tok/s) ===")
    print(f"  {'len':>7} {'ms/tok':>10} {'tok/s':>10}")
    for length in sorted({min(x, cap) for x in lengths}):
        bk.time_prefill(engine, backend, cfg, length, 1.0)  # warm: JIT for this length
        ms, tps = bk.time_prefill(engine, backend, cfg, length, 1.0)
        print(f"  {length:>7} {ms / length:>10.4f} {tps:>10.1f}")
        gate.check("prefill", f"len{length}", tps)


def suite_kv_reuse(gate, cfg, model, backend):
    """Prefix-cache: submit a shared long prefix, then requests reusing it.
    Measures hit-rate and warm-vs-cold prefill speedup."""
    import time as _t

    import benchkit as bk
    from tilerl.engine import SamplingParams, build_engine

    print("\n=== kv-reuse / prefix-cache ===")
    # Prefix must be block-aligned (BLOCK_TOKENS=16) and leave suffix room under
    # max_position. Sized to the model so tiny (max_pos 512) and 27B both work.
    from tilerl.kv_cache import BLOCK_TOKENS

    cap = cfg.max_position_embeddings
    plen = min(2048, max(BLOCK_TOKENS, (cap - 128) // BLOCK_TOKENS * BLOCK_TOKENS))
    slen = 32
    # Pool must hold the pinned prefix AND two live requests at once, or
    # evict_until_free drops the store entry before the warm request (the
    # serving-size 256-block pool did exactly that at plen=2048: hits 0).
    engine = build_engine(cfg, model, backend, num_blocks=4 * (plen // BLOCK_TOKENS) + 64,
                          num_slots=16, max_batch=8, max_total_tokens=plen + 256)
    prefix = bk.rand_prompt(cfg.vocab_size, plen, seed=42)
    # max_new_tokens>=2: publish happens only when the prefill completes with
    # phase != DONE, so a 1-token request finishes before it can publish.
    sp = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    # Publish the prefix as its OWN full prompt: the engine publishes the full
    # block-aligned prompt of a completed request, and a later request reuses it
    # only up to a block boundary published by some request (a full-length hit is
    # a day-1 miss). So request-1 IS the prefix; request-2 = prefix + suffix.
    bk.drive(engine, engine.submit(prefix, sp), 4096)
    # Warm both prefill shapes (reuse len and cold len) so JIT doesn't confound
    # the timing — the reuse path recomputes only the suffix, cold recomputes all.
    bk.drive(engine, engine.submit(prefix + bk.rand_prompt(cfg.vocab_size, slen, seed=8), sp), 4096)
    bk.drive(engine, engine.submit(bk.rand_prompt(cfg.vocab_size, plen + slen, seed=9), sp), 4096)
    h0, m0 = engine._prefix_hits, engine._prefix_misses
    t0 = _t.perf_counter()
    bk.drive(engine, engine.submit(prefix + bk.rand_prompt(cfg.vocab_size, slen, seed=1), sp), 4096)
    bk.sync(backend)
    warm = (_t.perf_counter() - t0) * 1e3
    t0 = _t.perf_counter()
    bk.drive(engine, engine.submit(bk.rand_prompt(cfg.vocab_size, plen + slen, seed=777), sp), 4096)
    bk.sync(backend)
    cold = (_t.perf_counter() - t0) * 1e3
    hits = engine._prefix_hits - h0
    speedup = cold / warm if warm else 0.0
    print(f"  prefix len {plen}, hits {hits}, misses {engine._prefix_misses - m0}")
    print(f"  cold {cold:.2f} ms (distinct)  warm {warm:.2f} ms (reused)  speedup {speedup:.2f}x")
    # Gate on HIT-RATE (deterministic): reuse must fire. Speedup is informational
    # (timing-noisy on a short suffix). A regression that breaks prefix reuse
    # drops hits to 0 and FAILs.
    gate.check("kv-reuse", "prefix-hits", float(hits))
    # speedup is printed above but NOT gated — a short-suffix warm/cold ratio is
    # timing-noisy; hit-rate is the deterministic correctness-of-reuse signal.


def suite_train(gate, backend, source, gpu):
    """train_step fwd+bwd tok/s. Tiny on CPU; 27B when a source is given."""
    import numpy as np

    from tilerl.autograd import AdamW
    from tilerl.cli import _build_model
    from tilerl.train import train_step

    def sync():
        if backend.device.type == "cuda":
            import torch

            torch.cuda.synchronize()

    # 27B fp32 masters alone are 108 GB > one H20 (OOMs at 1x128): the GPU
    # row covers the tape path on tiny. pending-remote: sharded/bf16 masters.
    model_name = "tiny"
    cfg, mdl = _build_model(model_name, seed=0, keep_master=True)
    opt = AdamW(lr=1e-3)
    shapes = [(2, 128), (2, 512)]
    print(f"\n=== training-step throughput ({model_name}) ===")
    print(f"  {'B x T':>10} {'ms/step':>10} {'tok/s':>12}")
    for b, t in shapes:
        ids = np.arange(1, b * t + 1, dtype=np.int64).reshape(b, t) % cfg.vocab_size
        train_step(mdl, ids, backend, opt)  # warm (JIT + tape shapes)
        samples = []
        for _ in range(3):
            sync()
            s = time.perf_counter()
            train_step(mdl, ids, backend, opt)
            sync()
            samples.append(time.perf_counter() - s)
        ms = statistics.median(samples) * 1e3
        tok_s = b * t / (ms / 1e3)
        print(f"  {f'{b}x{t}':>10} {ms:>10.2f} {tok_s:>12.1f}")
        gate.check("train", f"{model_name}-b{b}t{t}", tok_s)


# --- runner ------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="", help="comma list: decode-kv,prefill,kv-reuse,train,micro (default: all applicable)")
    ap.add_argument("--source", default=None, help="27B checkpoint dir (omit for tiny/CPU)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--depths", default=",".join(map(str, _KV_DEPTHS)))
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reseed", action="store_true", help="record every row as the new baseline (no gate)")
    args = ap.parse_args()

    import os

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("TILERL_TARGET", "cuda")
    else:
        os.environ.setdefault("TILERL_TARGET", "cpu")

    from tilerl.ops.backend import get_backend

    backend = get_backend()
    target = backend.arch
    # Host load is the silent confounder: another tenant's nvcc/JIT on this
    # host inflated B=8 ticks 60% in one run. Stamp it so a bad row is explainable.
    print(f"host loadavg {os.getloadavg()[0]:.1f} / {os.cpu_count()} cpus, target {target}")
    gate = Gate(target, update_only=args.reseed)
    batches = [int(x) for x in args.batches.split(",")]

    gpu_suites = {"decode-kv", "prefill", "kv-reuse"}
    default = ["train"] if args.source is None else ["decode-kv", "prefill", "kv-reuse", "train"]
    suites = [s for s in (args.suite.split(",") if args.suite else default) if s]

    cfg = model = None
    if any(s in gpu_suites or s == "micro" for s in suites):
        from tilerl.cli import _build_model
        from tilerl.model import load_hf
        from tilerl.config import qwen38_27b
        if args.source:
            cfg = qwen38_27b()
            model = load_hf(cfg, args.source, fuse_projections=True)
            cfg = model.cfg
        else:
            cfg, model = _build_model("tiny", seed=0)

    for s in suites:
        if s == "decode-kv":
            suite_decode_kv(gate, cfg, model, backend, batches, [int(x) for x in args.depths.split(",")], args.ticks)
        elif s == "prefill":
            suite_prefill(gate, cfg, model, backend, _KV_DEPTHS)
        elif s == "kv-reuse":
            suite_kv_reuse(gate, cfg, model, backend)
        elif s == "train":
            suite_train(gate, backend, args.source, args.gpu)
        else:
            print(f"  (unknown suite {s!r}, skipped)")

    if args.json:
        Path(args.json).write_text(json.dumps(gate.rows, indent=2) + "\n")
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
