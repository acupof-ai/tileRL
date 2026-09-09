"""tilerl bench harness: one runner, a snapshot baseline gate.

  decode-kv   decode tok/s vs KV depth at B=1..8
  prefill     prefill tok/s vs prompt length
  kv-reuse    prefix-cache hits + warm-vs-cold prefill
  spec        speculative decode goodput vs plain, same process (27B only)
  train       train_step fwd+bwd tok/s (tiny on CPU, 27B LoRA on the pod)
  train-full  full-parameter: bf16 masters, Adafactor; its own process (masters + fp4 base don't fit one card)
  accuracy    MMLU 0-shot % on a fixed slice (27B only) — the one non-speed gate

Baseline docs/experience/wins/bench-baseline.json, keyed (suite, shape, target) -> tok/s + commit
+ date, SOTA-only: one row per key, the best measurement. PASS at >= 0.97x, FAIL (exit 1)
below, and a beat is written as a CANDIDATE for review rather than promoted -- a run that
raises its own baseline cannot then regress against it. A first run seeds a missing key.

  uv run tilerl bench --suite train                       # CPU, tiny
  tilerl bench --source /work/Qwen3.8-27B-NVFP4 --gpu 7  # pod, all GPU suites
"""

from __future__ import annotations

import json
import os
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
        # .strip() or "unknown", not exists(): a redirect truncates the file before git
        # runs, so a failed stamp leaves it empty and a blank commit reads as provenance.
        return (stamp.read_text().strip() if stamp.exists() else "") or "unknown"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def _load_baseline() -> dict:
    if _BASELINE.exists():
        return json.loads(_BASELINE.read_text())
    return {}


def _save_baseline(b: dict) -> None:
    _BASELINE.write_text(json.dumps(b, indent=2, sort_keys=True) + "\n")
    # The pod's tree is wiped on every sync, so a row written only here never reaches
    # `pull`. Merged, not written: a plain write drops what another session raised meanwhile.
    shared = Path(os.environ.get("POD_BASELINE_DIR", "/work/tilerl-baseline"))
    if shared.is_dir():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from baseline import merge_into
        merge_into(shared / "bench-baseline.json", b)


class Gate:
    def __init__(self, target: str, update_only: bool = False):
        self.target = target
        self.baseline = _load_baseline()
        self.commit = _git_commit()
        self.date = _today()
        self.rows: list[dict] = []
        self.candidates: dict[str, dict] = {}
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
            # Proposed, not written: the json is the SOTA row the 0.97x gate compares
            # against, so a run that promotes its own result has nothing left to regress
            # against -- the next slow run is measured against the fast one it replaced.
            print(f"  BEAT   {key}: {prev['tok_s']:.1f} -> {tok_s:.1f} {unit}  "
                  f"(candidate, not written)")
            self.candidates[key] = {"tok_s": tok_s, "commit": self.commit, "date": self.date,
                                    "replaces": prev}
            verdict = "BEAT"
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

    def finish(self, candidate_path: Path | None = None) -> int:
        if self.dirty:  # seeds only: a key with no row yet has nothing to regress against
            _save_baseline(self.baseline)
            print(f"\n  baseline seeded: {_BASELINE}")
        if self.candidates and candidate_path is not None:
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps(self.candidates, indent=2, sort_keys=True) + "\n")
            print(f"  {len(self.candidates)} candidate row(s) for review: {candidate_path}")
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


def suite_prefill(gate, cfg, model, backend, lengths, build, model_name, device):
    import benchkit as bk
    import benchrec

    from tilerl.engine import build_engine

    cap = min(8192, cfg.max_position_embeddings)
    engine = build_engine(cfg, model, backend, num_blocks=cap // 16 + 64, num_slots=16,
                          max_batch=8, max_total_tokens=cap + 64)
    print("\n=== prefill-vs-length (tok/s) ===")
    print(f"  {'len':>7} {'ms/tok':>10} {'tok/s':>10} {'spread':>8}")
    for length in sorted({min(x, cap) for x in lengths}):
        bk.time_prefill(engine, backend, cfg, length, 1.0)  # JIT for this length, outside the window
        with benchrec.compiles_window(backend) as cw:
            runs = [bk.time_prefill(engine, backend, cfg, length, 1.0) for _ in range(3)]
        ms, tps = sorted(runs, key=lambda r: r[1])[1]
        spread = (max(r[1] for r in runs) - min(r[1] for r in runs)) / tps
        print(f"  {length:>7} {ms / length:>10.4f} {tps:>10.1f} {100 * spread:>7.1f}%")
        gate.check("prefill", f"len{length}", tps, spread=spread)
        floor = ({"value": 3835.0, "unit": "tok/s", "kind": "roofline",
                  "derivation": "3835 tok/s = mixed-dtype prefill roofline, fp4 attn 148 TFLOPS + "
                                "fp8 MLP 296 TFLOPS (wins/2026-08-24-sota-all-levers.md)"}
                 if device["name"] == "H20"
                 else {"value": round(tps, 1), "unit": "tok/s", "kind": "measured-best",
                       "derivation": "first accepted row for this population; floor = this measurement"})
        benchrec.append({
            "metric": "prefill_tok_s", "value": round(tps, 1), "unit": "tok/s",
            "target": backend.arch, "build": build, "model": model_name,
            "shape": {"ctx": length},
            "warm": {"state": "warm", "compiles": cw["compiles"]},
            "n": 3, "spread": round(spread, 4),
            "device": device, "commit": benchrec.git_commit(), "dirty": benchrec.git_dirty(),
            "cmd": " ".join(sys.argv),
            "floor": floor,
        })


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


def _gap(record: dict, registry: dict) -> float | None:
    """floor/value for higher-is-better, value/floor for lower-is-better.

    None when the value is 0 (a ratio has no floor there) -- such rows are
    instruments, not questions."""
    v = record["value"]
    if v <= 0:
        return None
    f = record["floor"]["value"]
    return f / v if registry[record["metric"]]["direction"] == "+" else v / f


_KIND_SHORT = {"bandwidth": "bw", "compute": "compute", "roofline": "roof",
               "measured-best": "best", "baseline": "base"}

#: A new best this far beyond the previous one is implausible until explained.
#: Run-to-run spread is ~1.7% (_RAISE), so 1.2x is ~10x noise: a real jump
#: gets an "explain" line, which is the right response to a real jump too.
_NEW_BEST_ALARM = 1.2


def _coverage(metrics: dict, rows) -> str:
    """One line every view prints: an empty store must make noise, not read as
    a clean bill of health."""
    measured = {r["metric"] for r in rows}
    return f"{len(metrics)} metrics declared, {len(measured)} measured"


def _view_table() -> None:
    """Four-target matrix; an empty cell says so, never a silent skip. The gap
    column carries its floor kind: roof/bw/compute/base are headroom against a
    physical limit, best is standing against our own (see --regress)."""
    import benchrec

    reg = benchrec.load_registry()
    metrics, denom = reg["metrics"], reg["denominator"]
    cur = benchrec.current(benchrec.load_all())
    groups: dict = {}
    for r in cur.values():
        g = (r["metric"], tuple(sorted(r["shape"].items())), r["build"])
        groups.setdefault(g, {})[r["target"]] = r
    print(f"=== bench table — denominator: a {denom['turn_s']}s agent turn = "
          f"{denom['prefill_s']}s prefill + {denom['decode_s']}s decode ({denom['source']}) ===")
    print(f"  coverage: {_coverage(metrics, cur.values())}, "
          f"{len({r['target'] for r in cur.values()})}/{len(benchrec.TARGETS)} targets covered")
    print(f"  {'metric (shape) [build]':<46} {'weight':>6} "
          f"{'cpu':>8} {'metal':>8} {'sm90':>8} {'sm70':>8} {'gap x w':>12}")
    missing = []
    for (metric, shape, build), cells in sorted(groups.items()):
        vals = []
        for t in benchrec.TARGETS:
            cell = cells.get(t)
            vals.append(f"{cell['value']:.1f}" if cell else "—")
            if cell is None:
                missing.append(f"{metric}{dict(shape)}/{t}")
        labeled = [(g, c["floor"]["kind"]) for t in benchrec.TARGETS
                   if (c := cells.get(t)) and (g := _gap(c, metrics))]
        if labeled:
            g, kind = max(labeled)
            gw = f"{g * metrics[metric]['weight']:.3f} {_KIND_SHORT.get(kind, kind)}"
        else:
            gw = "—"
        print(f"  {metric + ' ' + str(dict(shape)) + ' [' + build + ']':<46} "
              f"{metrics[metric]['weight']:>6.2f} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {gw:>12}")
    if missing:
        print("  empty cells (no accepted row): " + ", ".join(sorted(missing)))
    print("  gap kinds: roof/bw/compute/base = headroom vs a physical floor; "
          "best = vs our own best (a regression number, see --regress)")


def _view_readme() -> None:
    """The generated README rows: reuse speedup (turn 1 / turn 2 wall) and SSD restart."""
    import benchrec

    reg = benchrec.load_registry()["metrics"]
    cur = benchrec.current(benchrec.load_all())
    # HTML comment: paste-safe, but an empty store still makes noise.
    print(f"<!-- coverage: {_coverage(reg, cur.values())} -->")
    runs: dict = {}
    for r in cur.values():
        if r["metric"] != "chat_turn_wall_s":
            continue
        k = (r["cmd"], r["device"]["name"], r["build"], r["model"])
        runs.setdefault(k, {})[r["shape"].get("turn")] = r
    for turns in runs.values():
        if 0 in turns and 1 in turns:
            v1, v2 = turns[0]["value"], turns[1]["value"]
            print(f"| Cross-turn prefix reuse, turn 2 vs turn 1 | {v1 / v2:.1f}x "
                  f"| {turns[1]['device']['name']}, {turns[1]['build']}, record {turns[1]['id']} |")
    for r in cur.values():
        if r["metric"] == "ssd_restart_speedup":
            print(f"| SSD tier restart, warm vs cold | {r['value']:.3f}x "
                  f"| {r['device']['name']}, record {r['id']} |")


def _view_regress() -> None:
    """Two regression questions, kept apart: newest vs previous per population
    (n>=2 only — a point estimate with no dispersion makes no regression claim),
    and current rows standing below their population's best (floor.kind ==
    'measured-best'; gap > 1.0 means a better measurement exists in the store,
    FAIL past 1.05)."""
    import benchrec

    reg = benchrec.load_registry()["metrics"]
    cur = benchrec.current(benchrec.load_all())
    by_key: dict = {}
    for r in benchrec.load_all():
        if not benchrec.is_regressable(r):
            continue
        by_key.setdefault(benchrec.key(r), []).append(r)
    print("=== regression (newest vs previous, n>=2; PASS at >= 0.97x) ===")
    print(f"  coverage: {_coverage(reg, cur.values())}")
    for k, rows in sorted(by_key.items()):
        if len(rows) < 2:
            continue
        prev, last = rows[-2], rows[-1]
        d = reg[last["metric"]]["direction"]
        ratio = last["value"] / prev["value"] if d == "+" else prev["value"] / last["value"]
        print(f"  {'PASS' if ratio >= 0.97 else 'FAIL'} {last['metric']} {dict(last['shape'])} "
              f"{last['target']}/{last['build']}: {last['value']} vs {prev['value']} ({ratio:.3f}x)")

    print("=== vs our own best (measured-best floors; FAIL > 1.05x below best) ===")
    shown = 0
    for r in benchrec.current(benchrec.load_all()).values():
        if r["floor"]["kind"] != "measured-best":
            continue
        g = _gap(r, reg)
        if g is None:
            continue
        if g > 1.0:
            shown += 1
            print(f"  {'FAIL' if g > 1.05 else 'PASS'} {r['metric']} {dict(r['shape'])} "
                  f"{r['target']}/{r['build']}: {r['value']} vs best {r['floor']['value']} "
                  f"({g:.3f}x, n={r['n']})")
            continue
        # gap == 1.0: this row IS the best. First sight has no prior; a new best
        # that jumps far beyond the previous one is the other implausible shape
        # (too good, not too bad) — the symmetric half of the gate.
        prev = benchrec.previous_best(r, lower_is_better=reg[r["metric"]]["direction"] == "-")
        if prev is None:
            continue
        jump = prev / r["value"] if reg[r["metric"]]["direction"] == "-" else r["value"] / prev
        if jump > _NEW_BEST_ALARM:
            shown += 1
            print(f"  IMPLAUSIBLE {r['metric']} {dict(r['shape'])} "
                  f"{r['target']}/{r['build']}: {r['value']} vs previous best {prev} "
                  f"({jump:.2f}x, n={r['n']}) — explain or reject")
    if not shown:
        print("  (none — every measured-best row stands at its population's best)")


def _view_questions(limit: int = 20) -> None:
    """Headroom against physical floors only, by gap x weight desc. A
    measured-best gap is a regression, not headroom — see --regress; the two
    must not share a sorted column. Two louder todos rank above the list: a
    metric the registry declares but nobody has measured (worse than an
    unmeasured floor), and a metric with rows but no physical floor — the
    missing derivation is itself a todo. Both sorted by weight."""
    import benchrec

    reg = benchrec.load_registry()["metrics"]
    cur = list(benchrec.current(benchrec.load_all()).values())
    print(f"=== questions (headroom vs physical floor, gap x weight, top {limit}) ===")
    print(f"  coverage: {_coverage(reg, cur)}")
    # The loudest alarm first: a value that beats a hard physical floor is a
    # measurement error, not a result (135.5 tok/s vs a 129 roofline passed
    # every "good enough" gate for three days). Baseline floors are exempt —
    # the null is meant to be beaten.
    implausible = []
    unmeasured = sorted(
        ((reg[m]["weight"], m) for m in reg.keys() - {r["metric"] for r in cur}),
        reverse=True,
    )
    if unmeasured:
        print("=== no measurement at all ===")
        for w, m in unmeasured:
            print(f"  {m} (weight {w})")
    q = []
    floored: set = set()
    for r in cur:
        kind = r["floor"]["kind"]
        if kind not in benchrec.PHYSICAL_FLOOR_KINDS:
            continue
        floored.add(r["metric"])
        g = _gap(r, reg)
        if g is None:
            continue
        if g < 1.0 and kind in benchrec.HARD_FLOOR_KINDS:
            implausible.append((g, r))
        else:
            q.append((g * reg[r["metric"]]["weight"], g, r))
    if implausible:
        print("=== IMPLAUSIBLE — beat a physical floor: explain or reject ===")
        for g, r in sorted(implausible):
            print(f"  {r['metric']} {r['target']} {dict(r['shape'])}: "
                  f"{r['value']} vs {r['floor']['kind']} floor {r['floor']['value']} "
                  f"(gap {g:.3f}, record {r['id']})")
    q.sort(key=lambda x: x[0], reverse=True)
    for score, g, r in q[:limit]:
        print(f"  {score:.3f}  {r['metric']} {r['target']} {dict(r['shape'])} "
              f"[{r['floor']['kind']}]: {r['value']} vs floor {r['floor']['value']} "
              f"({g:.2f}x) x {reg[r['metric']]['weight']}")
    missing = sorted(
        ((reg[m]["weight"], m) for m in {r["metric"] for r in cur} - floored),
        reverse=True,
    )
    if missing:
        print("=== no physical floor — needs a derivation ===")
        for w, m in missing:
            print(f"  {m} (weight {w})")


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
    import types

    import benchrec
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
        with benchrec.compiles_window(backend) as cw:
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
        # Ruler record. Training engines never capture a graph or draft, so
        # build is "eager" by construction; the warmup step above covers JIT +
        # tape shapes, and compiles is the measured window delta (0 = clean).
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ns = types.SimpleNamespace(
            build="eager", target=backend.arch, device_name="",
            card=int(vis.split(",")[0]) if vis and backend.device.type == "cuda" else None,
            model_name=model_name)
        rec = {
            "metric": "train_step_tok_s", "value": round(tok_s, 1), "unit": "tok/s",
            "shape": {"batch": b, "ctx": t},
            "warm": {"state": "warm", "compiles": cw["compiles"]},
            "n": 3, "spread": round(spread, 4),
            **benchrec.record_common(ns),
        }
        rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=False)
        benchrec.append(rec)


def _view_collectors() -> None:
    """The metric -> collector map, and the gaps. A metric with no collector is a
    measurement nobody can reproduce by name; weight 0.94 rows with none are the
    batch queue."""
    import benchrec

    reg = benchrec.load_registry()["metrics"]
    print(f"{'metric':22s} {'w':>5s}  collector")
    for name, m in sorted(reg.items(), key=lambda kv: -kv[1]["weight"]):
        c = m.get("collector")
        if c:
            req = (" " + " ".join(c["required"])) if c["required"] else ""
            print(f"{name:22s} {m['weight']:5.2f}  {c['script']}{req}")
        elif c is None and "why_no_collector" in m:
            print(f"{name:22s} {m['weight']:5.2f}  -- (none: {m['why_no_collector']})")
        else:
            print(f"{name:22s} {m['weight']:5.2f}  MISSING")


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
    ap.add_argument("--candidates", default=None,
                    help="write rows that BEAT the baseline here for review; without it a "
                         "beat is printed and dropped, never written to the baseline")
    ap.add_argument("--spec-depth", type=int, default=2, help="drafts per row per tick")
    ap.add_argument("--mmlu-n", type=int, default=200, help="accuracy suite question count")
    ap.add_argument("--reseed", action="store_true", help="record every row as the new baseline (no gate)")
    ap.add_argument("--table", action="store_true", help="render the four-target table from the bench store and exit")
    ap.add_argument("--readme", action="store_true", help="render the README rows (reuse, SSD restart) and exit")
    ap.add_argument("--regress", action="store_true", help="regression diff vs previous per population and exit")
    ap.add_argument("--questions", action="store_true", help="rows by gap x weight desc and exit")
    ap.add_argument("--collectors", action="store_true", help="metric -> collector script map and exit")
    args = ap.parse_args()

    # Views are an explicit branch, never a fall-through: a query flag that misses
    # this list runs the training suite below and appends rows to the permanent store
    # (2026-09-09, --collectors did exactly that; the cmd field was the only tell).
    if args.table or args.readme or args.regress or args.questions or args.collectors:
        if args.table:
            _view_table()
        if args.readme:
            _view_readme()
        if args.regress:
            _view_regress()
        if args.questions:
            _view_questions()
        if args.collectors:
            _view_collectors()
        return 0

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

    # Record population for the bench store: load_hf fuses projections with a
    # --source, and prefill is not graph-captured (the graph is decode-only).
    if target in ("sm90", "sm70"):
        import torch

        bench_device = {"name": torch.cuda.get_device_name(0), "card": args.gpu}
    else:
        bench_device = {"name": target, "card": None}
    bench_build = "fused" if args.source else "eager"
    bench_model = "27B-nvfp4" if args.source else "tiny"

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
            suite_prefill(gate, cfg, model, backend, _KV_DEPTHS, bench_build, bench_model, bench_device)
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
    return gate.finish(Path(args.candidates) if args.candidates else None)


if __name__ == "__main__":
    sys.exit(main())
