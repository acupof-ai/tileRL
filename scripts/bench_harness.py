"""tilerl bench harness: one runner, a snapshot baseline gate.

  decode-kv   decode tok/s vs KV depth at B=1..8
  prefill     prefill tok/s vs prompt length
  kv-reuse    prefix-cache hits + warm-vs-cold prefill
  spec        speculative decode goodput vs plain, same process (27B only)
  train       train_step fwd+bwd tok/s (tiny on CPU, 27B LoRA on the pod)
  train-full  full-parameter: bf16 masters, Adafactor; its own process (masters + fp4 base don't fit one card)
  accuracy    MMLU 0-shot % on a fixed slice (27B only) — the one non-speed gate

Baseline docs/experience/wins/bench-baseline.json, keyed (suite, shape, target) -> tok/s + commit
+ date. PASS at >= 0.97x, auto-raise on a beat, FAIL (exit 1) below; a first run seeds it.

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
_GATE = 0.97
_RAISE = 1.02  # run-to-run spread is ~1.7%; raising on any overshoot ratchets the baseline on noise
_KV_DEPTHS = (512, 2048, 8192, 32768, 131072, 262144)  # 128K/256K: B=1 only (KV 17/34 GB)


_ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # the pod is a tarball, not a clone; pod_sync stamps HEAD here
        stamp = _ROOT / ".synced_commit"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load_baseline() -> dict:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


def _save_baseline(b: dict) -> None:
    _BASELINE.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")


class Gate:
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
            # A seed is held to the raise's noise bar, or a noisy row becomes a permanent baseline.
            if spread > _RAISE - 1.0:
                print(f"  NOISY {key}: {tok_s:.1f} {unit} at {100 * spread:.1f}% spread "
                      f"— not seeded")
                return "NOISY"
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "SEED"
        elif tok_s > prev["tok_s"] * _RAISE and spread <= _RAISE - 1.0:
            print(f"  RAISED {key}: {prev['tok_s']:.1f} -> {tok_s:.1f} tok/s")
            self.baseline[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date}
            self.dirty = True
            verdict = "RAISE"
        elif tok_s >= prev["tok_s"] * _GATE:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            if self.target != "cpu":  # CPU is ~4% noisy run-to-run: report-only
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


LAST_SPREAD = 0.0  # (max - min) / median of the last _median_windows call


def _median_windows(step_fn, n_windows: int, ticks: int, sync) -> float:
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


def suite_decode_kv(gate, cfg, model, backend, batches, depths, ticks):
    """Engine rebuilt per row so the pools are sized for that depth."""
    import benchkit as bk

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS

    _KV_BYTES_PER_TOKEN = 2 * len(cfg.full_attn_layers) * cfg.num_kv_heads * cfg.head_dim * 2
    _KV_BUDGET = 60 << 30  # 96 GiB card - 23 weights - working set

    print("\n=== decode-vs-KV-depth (tok/s, higher=better; should DROP with depth) ===")
    print(f"  {'depth':>7} {'B':>3} {'ms/tick':>9} {'tok/s/req':>10} {'agg tok/s':>10}"
          f" {'spread':>8}")
    for depth in depths:
        if depth > cfg.max_position_embeddings:
            continue
        for b in batches:
            # gen must outlive the staggered prefill (512/tick) or row 0 finishes before settle sees B rows.
            gen = ticks + 40 + b * (depth // 512 + 4)
            # 2x headroom (the prefix store pins finished prompts); 1.1x at 128K/256K where KV alone is 17/34 GB.
            head = 2 if depth <= 32768 else 1.1
            if depth * b * head * _KV_BYTES_PER_TOKEN > _KV_BUDGET:
                print(f"  {depth:>7} {b:>3}   (skipped: KV pool would exceed one H20)")
                continue
            need = int(head * (-(-(depth + gen) * b // BLOCK_TOKENS) + b))
            # The guard sizes the pool only; activations can still OOM, and that is a row, not the end of the run.
            try:
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
                done: dict = {}
                for _ in range(ticks + 8 * b + 256):
                    done.update(engine.poll())
                    if all(w in done for w in wids):
                        break
                    engine.step()
            except torch_oom() as exc:
                print(f"  {depth:>7} {b:>3} {'OOM':>9}  {str(exc).split('.')[0]}")
            finally:
                engine = wids = None
                _free(backend)


def suite_spec(gate, cfg, model, backend, batches, source, ticks, depth):
    """Gated as a ratio against the plain arm measured in the same process."""
    import benchkit as bk

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.spec import load_draft

    path = Path(source) / "model_mtp.safetensors"
    if not path.exists():
        print(f"\n  (no draft head at {path}, spec suite skipped)")
        return
    print(f"\n=== speculative decode (depth {depth}, tok/s) ===")
    print(f"  {'B':>3} {'arm':>6} {'ms/tick':>9} {'tok/tick':>9} {'accept':>7} {'agg tok/s':>10}")
    base: dict[int, float] = {}
    for b, spec in [(b, s) for b in batches for s in (False, True)]:
        engine = build_engine(
            cfg, model, backend, num_blocks=512, num_slots=max(16, b), max_batch=max(8, b),
            draft=load_draft(model, path) if spec else None, spec_depth=depth,
        )
        # A request finishing mid-window leaves ticks generating nothing and drags tok/tick down.
        budget = (bk.SETTLE_BUDGET(b) + 8 + 3 * ticks + 4) * (1 + depth)
        for i in range(b):
            engine.submit(
                bk.rand_prompt(cfg.vocab_size, 16, seed=700 + i),
                SamplingParams(temperature=0.0, max_new_tokens=budget, seed=i),
            )
        if not bk.settle_decode(engine, b, 64 + 8 * b):
            print(f"  {b:>3}   (never reached pure decode — skipped)")
            continue
        for _ in range(8):
            engine.step()
        bk.sync(backend)
        s0 = engine.stats()
        ms = _median_windows(engine.step, 3, ticks, lambda: bk.sync(backend))
        s1 = engine.stats()
        if len(engine._running) != b:
            print(f"  {b:>3} {'d' + str(depth) if spec else 'plain':>6}   "
                  f"(a request finished mid-window — budget too small, row void)")
            continue
        per_tick = (s1["tokens_generated"] - s0["tokens_generated"]) / (3 * ticks) / b
        drafted = s1["spec_drafted"] - s0["spec_drafted"]
        acc = (s1["spec_accepted"] - s0["spec_accepted"]) / max(drafted, 1)
        agg = 1000.0 * b * per_tick / ms
        arm = f"d{depth}" if spec else "plain"
        print(f"  {b:>3} {arm:>6} {ms:>9.3f} {per_tick:>9.2f} {100 * acc:>6.1f}% {agg:>10.1f}"
              + (f"   {agg / base[b]:.2f}x vs plain" if spec and base.get(b) else ""))
        if spec:
            gate.check("spec", f"d{depth}-b{b}-ratio", agg / base[b], spread=LAST_SPREAD)
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
        bk.time_prefill(engine, backend, cfg, length, 1.0)  # JIT for this length
        runs = [bk.time_prefill(engine, backend, cfg, length, 1.0) for _ in range(3)]
        ms, tps = sorted(runs, key=lambda r: r[1])[1]
        spread = (max(r[1] for r in runs) - min(r[1] for r in runs)) / tps
        print(f"  {length:>7} {ms / length:>10.4f} {tps:>10.1f} {100 * spread:>7.1f}%")
        gate.check("prefill", f"len{length}", tps, spread=spread)


def suite_kv_reuse(gate, cfg, model, backend):
    import benchkit as bk

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS

    print("\n=== kv-reuse / prefix-cache ===")
    cap = cfg.max_position_embeddings
    plen = min(2048, max(BLOCK_TOKENS, (cap - 128) // BLOCK_TOKENS * BLOCK_TOKENS))
    slen = 32
    # Pool holds the pinned prefix plus two live requests, or eviction drops the entry before the warm request.
    engine = build_engine(cfg, model, backend, num_blocks=4 * (plen // BLOCK_TOKENS) + 64,
                          num_slots=16, max_batch=8, max_total_tokens=plen + 256)
    prefix = bk.rand_prompt(cfg.vocab_size, plen, seed=42)
    # max_new_tokens>=2: a 1-token request finishes before its prefill can publish.
    sp = SamplingParams(temperature=0.0, max_new_tokens=2, seed=0)
    # The prefix is its own request: a later request reuses only up to a published block boundary.
    bk.drive(engine, engine.submit(prefix, sp), 4096)
    # Warm both prefill shapes (suffix-only and full) so JIT does not confound the timing.
    bk.drive(engine, engine.submit(prefix + bk.rand_prompt(cfg.vocab_size, slen, seed=8), sp), 4096)
    bk.drive(engine, engine.submit(bk.rand_prompt(cfg.vocab_size, plen + slen, seed=9), sp), 4096)
    h0, m0 = engine._prefix_hits, engine._prefix_misses
    t0 = time.perf_counter()
    bk.drive(engine, engine.submit(prefix + bk.rand_prompt(cfg.vocab_size, slen, seed=1), sp), 4096)
    bk.sync(backend)
    warm = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    bk.drive(engine, engine.submit(bk.rand_prompt(cfg.vocab_size, plen + slen, seed=777), sp), 4096)
    bk.sync(backend)
    cold = (time.perf_counter() - t0) * 1e3
    hits = engine._prefix_hits - h0
    speedup = cold / warm if warm else 0.0
    print(f"  prefix len {plen}, hits {hits}, misses {engine._prefix_misses - m0}")
    print(f"  cold {cold:.2f} ms (distinct)  warm {warm:.2f} ms (reused)  speedup {speedup:.2f}x")
    # Hits are deterministic; the short-suffix speedup is timing-noisy and not gated.
    gate.check("kv-reuse", "prefix-hits", float(hits), unit="hits")


def suite_accuracy(gate, source, n):
    """Greedy over a fixed slice, so any move at all is a real change."""
    from mmlu import accuracy

    print("\n=== accuracy (MMLU 0-shot) ===")
    correct, total = accuracy(source, n=n)
    pct = 100.0 * correct / total
    print(f"  {correct}/{total} = {pct:.1f}%")
    gate.check("accuracy", f"mmlu-{total}", pct, unit="%")


def torch_oom():
    import torch

    return getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def _free(backend) -> None:
    import gc

    import torch

    gc.collect()
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()


def suite_train(gate, backend, source, full=False):
    import numpy as np

    from tilerl.autograd import Adafactor, AdamW
    from tilerl.cli import _build_model
    from tilerl.train import train_step

    def sync():
        if backend.device.type == "cuda":
            import torch

            torch.cuda.synchronize()

    # Full-parameter 27B fits at 73.2 of 95 GiB: bf16 masters only, Adafactor (Adam's m+v is 200.4 GiB).
    trainable = None
    if source:
        from tilerl.config import qwen38_27b
        from tilerl.model import add_lora, drop_quantized, load_hf

        cfg = qwen38_27b()
        model_name = "27B-full" if full else "27B-lora"
        mdl = load_hf(cfg, source, fuse_projections=False, keep_master=full)
        if full:
            drop_quantized(mdl)
        mdl.params = backend.materialize(mdl.params)
        if not full:
            trainable = add_lora(mdl, rank=16)
        # A slope in T (peak GB/token decides recompute) and in B (the step is launch-bound, ~491K kernels).
        shapes = [(1, 64), (1, 128), (1, 256), (2, 256), (4, 256)]
    else:
        model_name = "tiny"
        cfg, mdl = _build_model(model_name, seed=0, keep_master=True)
        shapes = [(2, 128), (2, 512)]
    opt = Adafactor(lr=1e-2) if full else AdamW(lr=1e-3)
    print(f"\n=== training-step throughput ({model_name}) ===")
    print(f"  {'B x T':>10} {'ms/step':>10} {'tok/s':>12}")
    for b, t in shapes:
        ids = np.arange(1, b * t + 1, dtype=np.int64).reshape(b, t) % cfg.vocab_size
        try:
            train_step(mdl, ids, backend, opt, trainable=trainable)  # warm (JIT+tape shapes)
        except torch_oom() as exc:  # the shape that does not fit is a row, not the end of the run
            print(f"  {f'{b}x{t}':>10} {'OOM':>10}  {str(exc).split('.')[0]}")
            _free(backend)
            continue
        samples = []
        for _ in range(3):
            sync()
            s = time.perf_counter()
            train_step(mdl, ids, backend, opt, trainable=trainable)
            sync()
            samples.append(time.perf_counter() - s)
        ms = statistics.median(samples) * 1e3
        spread = (max(samples) - min(samples)) / statistics.median(samples)
        tok_s = b * t / (ms / 1e3)
        peak = ""
        if backend.device.type == "cuda":
            import torch

            peak = f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
            torch.cuda.reset_peak_memory_stats()
        print(f"  {f'{b}x{t}':>10} {ms:>10.2f} {tok_s:>12.1f}{peak}  +-{100 * spread:.1f}%")
        gate.check("train", f"{model_name}-b{b}t{t}", tok_s, spread=spread)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="",
                    help="comma list: decode-kv,prefill,kv-reuse,spec,train,train-full,"
                         "accuracy"
                         " (default: all applicable)")
    ap.add_argument("--source", default=None, help="27B checkpoint dir (omit for tiny/CPU)")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--batches", default="1,2,4,8")  # B=2 once cost a B=8 tick; a 1,8 sweep cannot see that
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
    # Host load stamped: another tenant's JIT once inflated B=8 ticks 60%.
    print(f"host loadavg {os.getloadavg()[0]:.1f} / {os.cpu_count()} cpus, target {target}")
    gate = Gate(target, update_only=args.reseed)
    batches = [int(x) for x in args.batches.split(",")]

    gpu_suites = {"decode-kv", "prefill", "kv-reuse", "spec"}
    default = (["train"] if args.source is None
               else ["decode-kv", "prefill", "kv-reuse", "spec", "train", "accuracy"])
    suites = [s for s in (args.suite.split(",") if args.suite else default) if s]

    cfg = model = None
    if any(s in gpu_suites for s in suites):
        from tilerl.cli import _build_model
        from tilerl.config import qwen38_27b
        from tilerl.model import load_hf
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
            suite_train(gate, backend, args.source)
        elif s == "train-full":
            suite_train(gate, backend, args.source, full=True)
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
