"""tilerl bench harness: one runner, five suites, a snapshot baseline gate.

Fills the coverage holes the scattered bench_*.py scripts left open:

  decode-kv   decode tok/s vs KV depth (512/2k/8k/32k) at B=1,8 — the serving
              metric that DEGRADES with context (paged-attention KV read scales
              with depth). The old verify check 4 timed one shallow ~512 depth.
  prefill     prefill tok/s vs prompt length, as a curve not 3 points.
  kv-reuse    prefix-cache hit-rate + warm-vs-cold prefill speedup (the engine's
              PrefixStore was perf-untested).
  spec        speculative decode goodput vs the draft head (27B only).
  train       train_step fwd+bwd tok/s (tiny on CPU, 27B on the pod) — training
              had zero perf coverage.
  accuracy    MMLU 0-shot % on a fixed slice (27B only). Every other suite gates
              speed; without this one an "optimization" that breaks the logits
              passes the whole harness.

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
#: A win must clear the measured run-to-run spread before it raises the
#: snapshot. Repeated d512-b1 readings on this pod land within ~1.7% of each
#: other, so a baseline that raises on ANY overshoot ratchets up on noise until
#: no run can meet it. Below this, a faster reading is a PASS, not a RAISE.
_RAISE = 1.02
_KV_DEPTHS = (512, 2048, 8192, 32768, 131072, 262144)  # 128K/256K: B=1 only (KV 17/34 GB)


_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        # The pod arrives as a tarball, not a clone; pod_sync stamps HEAD here.
        stamp = _ROOT / ".synced_commit"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


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

    def check(self, suite: str, shape: str, tok_s: float, unit: str = "tok/s",
              spread: float = 0.0) -> str:
        key = f"{suite}/{shape}/{self.target}"
        prev = self.baseline.get(key)
        verdict = "SEED"
        if prev is None or self.seed_only:
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "SEED"
        elif tok_s > prev["tok_s"] * _RAISE and spread <= _RAISE - 1.0:
            print(f"  RAISED {key}: {prev['tok_s']:.1f} -> {tok_s:.1f} tok/s")
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "RAISE"
        elif tok_s >= prev["tok_s"] * _GATE:
            verdict = "PASS"  # includes a win too noisy to raise the baseline
        else:
            verdict = "FAIL"
            # CPU is thermally/scheduler-noisy (~4% run-to-run) — gate it as a
            # report-only smoke, not a hard fail. The 3% gate is for the stable
            # GPU baseline (matches CI: only lint + CPU-correctness block).
            if self.target != "cpu":
                self.failed = True
        base = prev["tok_s"] if prev else tok_s
        self.rows.append(
            {"key": key, "tok_s": tok_s, "unit": unit, "baseline": base,
             "ratio": tok_s / base if base else 0.0, "verdict": verdict}
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
                f"  {r['verdict']:<5} {r['key']:<44} {r['tok_s']:>9.1f} {r['unit']:<5}"
                f"  ({r['ratio']:.3f}x){soft}"
            )
        return 1 if self.failed else 0


# --- timing (median of windows, reusing verify's steady-state settle) --------


#: Spread of the last _median_windows call: (max - min) / median. A row whose
#: spread is wide is not evidence about a few-percent change, whatever its
#: median says — printed beside every timing so a reader can judge the verdict.
LAST_SPREAD = 0.0


def _median_windows(step_fn, n_windows: int, ticks: int, sync) -> float:
    """Median ms/tick over n_windows of `ticks` steady-state steps each."""
    global LAST_SPREAD
    samples = []
    for _ in range(n_windows):
        sync()
        t0 = time.perf_counter()
        for _ in range(ticks):
            step_fn()
        sync()
        samples.append((time.perf_counter() - t0) / ticks * 1e3)
    med = statistics.median(samples)
    LAST_SPREAD = (max(samples) - min(samples)) / med if med else 0.0
    return med


# --- suites ------------------------------------------------------------------


def suite_decode_kv(gate, cfg, model, backend, batches, depths, ticks):
    """Decode tok/s vs KV depth. Rebuilds the engine per depth so the pools are
    sized for that KV (32k*B tokens needs many blocks) — settle_decode drives
    every row to pure-decode before timing."""
    import benchkit as bk
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS

    print("\n=== decode-vs-KV-depth (tok/s, higher=better; should DROP with depth) ===")
    print(f"  {'depth':>7} {'B':>3} {'ms/tick':>9} {'tok/s/req':>10} {'agg tok/s':>10}"
          f" {'spread':>8}")
    for depth in depths:
        if depth > cfg.max_position_embeddings:
            continue
        for b in batches:
            # 2x headroom: the prefix store pins every finished prompt's blocks
            # for reuse, so a run that just drained still holds ~depth*B blocks.
            # outlive the batch's staggered prefill (chunks of 512/tick) or row 0 is
            # DONE before the last row reaches decode and settle never sees B rows.
            gen = ticks + 40 + b * (depth // 512 + 4)
            # 2x headroom (the prefix store pins finished prompts) up to 32K; at
            # 128K/256K the KV alone is 17/34 GB, so 1.1x and one row per pool.
            head = 2 if depth <= 32768 else 1.1
            if depth * b * head > 294912:  # ~38 GB of KV + 23 GB weights on a 96 GB H20
                print(f"  {depth:>7} {b:>3}   (skipped: KV pool would exceed one H20)")
                continue
            need = int(head * (-(-(depth + gen) * b // BLOCK_TOKENS) + b))
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
            print(f"  {depth:>7} {b:>3} {ms:>9.3f} {1000.0 / ms:>10.1f} {agg:>10.1f}"
                  f" {100 * LAST_SPREAD:>7.1f}%")
            gate.check("decode-kv", f"d{depth}-b{b}", agg, spread=LAST_SPREAD)
            # drain so the next depth's blocks are free
            done: dict = {}
            for _ in range(ticks + 8 * b + 256):
                done.update(engine.poll())
                if all(w in done for w in wids):
                    break
                engine.step()


def suite_spec(gate, cfg, model, backend, batches, source, ticks, depth):
    """Speculative decode goodput. Gates the AGGREGATE tok/s, which is what
    speculation is for; acceptance is printed because a drop there is the first
    sign the draft head or its serving format regressed."""
    import benchkit as bk
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.spec import load_draft

    path = Path(source) / "model_mtp.safetensors"
    if not path.exists():
        print(f"\n  (no draft head at {path}, spec suite skipped)")
        return
    print(f"\n=== speculative decode (depth {depth}, tok/s) ===")
    print(f"  {'B':>3} {'arm':>6} {'ms/tick':>9} {'tok/tick':>9} {'accept':>7} {'agg tok/s':>10}")
    # Both arms measured here, identically: comparing a spec row against a
    # decode row from another script is what made a 4.9x LOSS read as a 1.14x
    # win this morning (that one was eager-vs-graph; this one would be
    # settled-vs-not).
    base: dict[int, float] = {}
    for b, spec in [(b, s) for b in batches for s in (False, True)]:
        engine = build_engine(
            cfg, model, backend, num_blocks=512, num_slots=max(16, b), max_batch=max(8, b),
            draft=load_draft(model, path) if spec else None, spec_depth=depth,
        )
        for i in range(b):  # noqa: B007 - both arms share the prompt set
            engine.submit(
                bk.rand_prompt(cfg.vocab_size, 16, seed=700 + i),
                SamplingParams(temperature=0.0, max_new_tokens=(ticks + 20) * (1 + depth), seed=i),
            )
        # One waiting request is admitted per tick, so B rows need at least B
        # ticks before any of them decodes: a fixed warmup times prefill ticks
        # as if they were decode (B=8 read 51.8 tok/s against 91.8 settled).
        if not bk.settle_decode(engine, b, 64 + 8 * b):
            print(f"  {b:>3}   (never reached pure decode — skipped)")
            continue
        for _ in range(8):
            engine.step()
        bk.sync(backend)
        s0 = engine.stats()
        ms = _median_windows(engine.step, 3, ticks, lambda: bk.sync(backend))
        s1 = engine.stats()
        per_tick = (s1["tokens_generated"] - s0["tokens_generated"]) / (3 * ticks) / b
        drafted = s1["spec_drafted"] - s0["spec_drafted"]
        acc = (s1["spec_accepted"] - s0["spec_accepted"]) / max(drafted, 1)
        agg = 1000.0 * b * per_tick / ms
        arm = f"d{depth}" if spec else "plain"
        print(f"  {b:>3} {arm:>6} {ms:>9.3f} {per_tick:>9.2f} {100 * acc:>6.1f}% {agg:>10.1f}"
              + (f"   {agg / base[b]:.2f}x vs plain" if spec and base.get(b) else ""))
        if spec:
            gate.check("spec", f"d{depth}-b{b}", agg, spread=LAST_SPREAD)
        else:
            base[b] = agg


def suite_prefill(gate, cfg, model, backend, lengths):
    import benchkit as bk
    from tilerl.engine import build_engine

    cap = min(8192, cfg.max_position_embeddings)
    engine = build_engine(cfg, model, backend, num_blocks=cap // 16 + 64, num_slots=16,
                          max_batch=8, max_total_tokens=cap + 64)
    print("\n=== prefill-vs-length (tok/s) ===")
    print(f"  {'len':>7} {'ms/tok':>10} {'tok/s':>10} {'spread':>8}")
    for length in sorted({min(x, cap) for x in lengths}):
        bk.time_prefill(engine, backend, cfg, length, 1.0)  # warm: JIT for this length
        # Three readings, not one: prefill has no steady-state loop to hide
        # run-to-run noise, and a single sample can neither be trusted nor
        # allowed to ratchet the baseline.
        runs = [bk.time_prefill(engine, backend, cfg, length, 1.0) for _ in range(3)]
        ms, tps = sorted(runs, key=lambda r: r[1])[1]
        spread = (max(r[1] for r in runs) - min(r[1] for r in runs)) / tps
        print(f"  {length:>7} {ms / length:>10.4f} {tps:>10.1f} {100 * spread:>7.1f}%")
        gate.check("prefill", f"len{length}", tps, spread=spread)


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
    gate.check("kv-reuse", "prefix-hits", float(hits), unit="hits")
    # speedup is printed above but NOT gated — a short-suffix warm/cold ratio is
    # timing-noisy; hit-rate is the deterministic correctness-of-reuse signal.


def suite_accuracy(gate, source, n):
    """MMLU 0-shot on a fixed slice — the only gate here that is not a speed gate.

    Greedy over a fixed question set, so the number is deterministic: any move
    at all is a real change, not sampling noise.
    """
    from mmlu import accuracy

    print("\n=== accuracy (MMLU 0-shot) ===")
    correct, total = accuracy(source, n=n)
    pct = 100.0 * correct / total
    print(f"  {correct}/{total} = {pct:.1f}%")
    gate.check("accuracy", f"mmlu-{total}", pct, unit="%")


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

    # Full-parameter 27B needs 54 GB of bf16 masters + 216 GB of Adam moments
    # and does not fit one H20 at any shape. The source row trains LoRA
    # adapters on the frozen fp4 base instead (model.add_lora); tiny keeps the
    # full-parameter tape covered on CPU.
    trainable = None
    if source:
        from tilerl.config import qwen38_27b
        from tilerl.model import add_lora, load_hf

        model_name = "27B-lora"
        cfg = qwen38_27b()
        mdl = load_hf(cfg, source, fuse_projections=False)
        mdl.params = backend.materialize(mdl.params)
        trainable = add_lora(mdl, rank=16)
        # a slope, not one point: the tape keeps every activation in f32, so
        # peak GB per token is the number that decides whether recompute is needed
        shapes = [(1, 64), (1, 128), (1, 256)]
    else:
        model_name = "tiny"
        cfg, mdl = _build_model(model_name, seed=0, keep_master=True)
        shapes = [(2, 128), (2, 512)]
    opt = AdamW(lr=1e-3)
    print(f"\n=== training-step throughput ({model_name}) ===")
    print(f"  {'B x T':>10} {'ms/step':>10} {'tok/s':>12}")
    for b, t in shapes:
        ids = np.arange(1, b * t + 1, dtype=np.int64).reshape(b, t) % cfg.vocab_size
        train_step(mdl, ids, backend, opt, trainable=trainable)  # warm (JIT + tape shapes)
        samples = []
        for _ in range(3):
            sync()
            s = time.perf_counter()
            train_step(mdl, ids, backend, opt, trainable=trainable)
            sync()
            samples.append(time.perf_counter() - s)
        ms = statistics.median(samples) * 1e3
        tok_s = b * t / (ms / 1e3)
        peak = ""
        if backend.device.type == "cuda":
            import torch

            peak = f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
            torch.cuda.reset_peak_memory_stats()
        print(f"  {f'{b}x{t}':>10} {ms:>10.2f} {tok_s:>12.1f}{peak}")
        gate.check("train", f"{model_name}-b{b}t{t}", tok_s)


# --- runner ------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="",
                    help="comma list: decode-kv,prefill,kv-reuse,spec,train,accuracy"
                         " (default: all applicable)")
    ap.add_argument("--source", default=None, help="27B checkpoint dir (omit for tiny/CPU)")
    ap.add_argument("--gpu", type=int, default=None)
    # 2 and 4 are not decoration: mma8 pads M to 8, so B=2 cost the same tick as
    # B=8 and was SLOWER in aggregate than B=1 — invisible to a 1,8 sweep
    # (errors/2026-08-29-spec-cost-was-the-linear-not-the-draft.md).
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--depths", default=",".join(map(str, _KV_DEPTHS)))
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--spec-depth", type=int, default=2, help="drafts per row per tick")
    ap.add_argument("--mmlu-n", type=int, default=200, help="accuracy suite question count")
    ap.add_argument("--reseed", action="store_true", help="record every row as the new baseline (no gate)")
    args = ap.parse_args()

    import os

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("TILERL_TARGET", "cuda")
    else:
        os.environ.setdefault("TILERL_TARGET", "cpu")

    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    target = backend.arch
    # Host load is the silent confounder: another tenant's nvcc/JIT on this
    # host inflated B=8 ticks 60% in one run. Stamp it so a bad row is explainable.
    print(f"host loadavg {os.getloadavg()[0]:.1f} / {os.cpu_count()} cpus, target {target}")
    gate = Gate(target, update_only=args.reseed)
    batches = [int(x) for x in args.batches.split(",")]

    gpu_suites = {"decode-kv", "prefill", "kv-reuse", "spec"}
    default = (["train"] if args.source is None
               else ["decode-kv", "prefill", "kv-reuse", "spec", "train", "accuracy"])
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
        elif s == "spec":
            if args.source is None:
                print("  (spec needs --source, skipped)")
            else:
                suite_spec(gate, cfg, model, backend, batches, args.source, args.ticks,
                           args.spec_depth)
        elif s == "train":
            suite_train(gate, backend, args.source, args.gpu)
        elif s == "accuracy":
            if args.source is None:
                print("  (accuracy needs --source, skipped)")
            else:
                suite_accuracy(gate, args.source, args.mmlu_n)
        else:
            print(f"  (unknown suite {s!r}, skipped)")

    if args.json:
        Path(args.json).write_text(json.dumps(gate.rows, indent=2) + "\n")
    return gate.finish()


if __name__ == "__main__":
    sys.exit(main())
