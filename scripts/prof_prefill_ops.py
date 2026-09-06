"""One prefill, decomposed into measured seconds per op. Which kernel carries the n^2?

TTFT on the V100 fits ``0.56 + 0.00422*n + 4.117e-07*n^2`` (R^2 0.9999,
docs/experience/wins/2026-09-07-v100-prefill-is-quadratic-not-per-chunk.md), and
the n^2 term is 68% of a 21.7k-token prompt. That measurement says the cost is
per token rather than per 512-token chunk; it does not say which op spends it.
This does.

**How the timing works.** Every op the model calls goes through ``backend.<name>``
(model.py calls nothing else), so a proxy around the Backend times all of them
without touching model.py or engine.py. The proxy is this script's, not the
runtime's -- nothing here changes what ships.

**Why per chunk, and why prefix_from matters.** A prompt over the token budget
prefills in 512-token chunks, and chunk i attends over the whole prefix
materialized so far, not over its own 512 tokens. Cost per chunk therefore RISES
across a single prompt, and a profile that sums the chunks together cannot show
it. Each chunk is recorded with the prefix length it ran against, so the per-op
series can be fitted against prefix length -- an op whose seconds rise with the
prefix carries the n^2; an op flat across chunks carries the linear term.

**Device time.** With CUDA the clock is read after ``torch.cuda.synchronize()``,
so each op's seconds are device-inclusive. The syncs are the instrument's, they
serialize what would otherwise overlap, and their own cost is reported
(``sync_secs``) so a total can be reconciled against an unsynced run.

**Maker provenance.** For each op the profile prints which factory the registry
resolved and from which module, per arch. sm70 has previously registered a
CPU-authored kernel under a name that reads as arch-specific, so the name alone
does not say what ran.

    # CPU gate (this machine)
    TILERL_TARGET=cpu uv run python scripts/prof_prefill_ops.py --selfcheck

    # V100, inside the maintenance window, with the serve child stopped
    /work/tl013/bin/python -u scripts/prof_prefill_ops.py \\
        --model qwen38-27b --tokens 2048,8192,16384 --json /work/prefill_ops.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "packages/tilerl-kernels/src")

from tilerl_kernels.backend import get_backend  # noqa: E402
from tilerl_kernels.registry import _REGISTRY  # noqa: E402

from tilerl.config import tiny  # noqa: E402
from tilerl.engine import _PHASE_PREFILL, SamplingParams, build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402

_MAX_TICKS = 20000

#: ops worth a row of their own; everything else lands in "other" rather than
#: being dropped, so the buckets sum to the tick.
_WATCH = (
    "paged_attention", "attention", "attn_prep", "write_tokens",
    "linear", "linear_fp4", "linear_fp8",
    "gdn_prep", "gdn_post", "linear_attn_chunk", "gdn_span_ab_raw",
    "rmsnorm", "rmsnorm_f32", "rope", "silu_mul", "softmax", "embedding", "add",
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timer:
    """Times every backend call by name. Attribute access, not a subclass: the
    engine holds one Backend and passes it down, so wrapping the instance is the
    only way to see calls made from inside model.py."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "secs", defaultdict(float))
        object.__setattr__(self, "calls", defaultdict(int))
        object.__setattr__(self, "sync_secs", 0.0)
        object.__setattr__(self, "on", False)

    def __getattr__(self, name):
        attr = getattr(object.__getattribute__(self, "_inner"), name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def timed(*a, **kw):
            if not object.__getattribute__(self, "on"):
                return attr(*a, **kw)
            t0 = time.perf_counter()
            out = attr(*a, **kw)
            t1 = time.perf_counter()
            _sync()
            t2 = time.perf_counter()
            s = object.__getattribute__(self, "secs")
            c = object.__getattribute__(self, "calls")
            s[name] += t2 - t0
            c[name] += 1
            object.__setattr__(self, "sync_secs",
                               object.__getattribute__(self, "sync_secs") + (t2 - t1))
            return out

        return timed

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def reset(self):
        object.__getattribute__(self, "secs").clear()
        object.__getattribute__(self, "calls").clear()
        object.__setattr__(self, "sync_secs", 0.0)

    def snapshot(self):
        return dict(object.__getattribute__(self, "secs")), \
            dict(object.__getattribute__(self, "calls"))


def _provenance(precision: str) -> dict:
    """Which factory backs each watched op, per arch, and from which module.

    A shared NAME is not a shared kernel and not a distinct one either -- resolve
    the maker. sm70's paged_attention is `kernels.make_paged_attention`, the same
    object the cpu cell registers, while sm90's is `kernels_attn.make_..._mma`.
    """
    out = {}
    for arch in ("cpu", "sm70", "sm90", "metal"):
        cell = _REGISTRY.get((precision, arch))
        if not cell:
            continue
        out[arch] = {
            name: f"{getattr(fn, '__module__', '?').rsplit('.', 1)[-1]}."
                  f"{getattr(fn, '__name__', '<lambda>')}"
            for name, fn in sorted(cell.items())
        }
    return out


def _prefill_arm(engine, timer, n_tokens: int, vocab: int, seed: int) -> dict:
    """Submit one prompt and step until its prefill is done, timing each tick.

    A tick is attributed to the chunk it prefilled by reading the engine's own
    `prefill_from` before the step: that is the prefix the chunk attended over,
    and it is what the per-op series is fitted against.
    """
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, vocab, (n_tokens,), generator=g).tolist()
    rid = engine.submit(ids, SamplingParams(max_new_tokens=1, temperature=0.0, seed=0))
    req = next(r for r in list(engine._running) + list(engine._waiting) if r.req_id == rid)

    chunks, ticks = [], 0
    _sync()
    t_arm = time.perf_counter()
    object.__setattr__(timer, "on", True)
    while req.phase == _PHASE_PREFILL and ticks < _MAX_TICKS:
        prefix = req.prefill_from
        timer.reset()
        _sync()
        t0 = time.perf_counter()
        engine.step()
        _sync()
        dt = time.perf_counter() - t0
        secs, calls = timer.snapshot()
        chunks.append({
            "prefix": prefix,
            "chunk": req.prefill_from - prefix,
            "tick_secs": dt,
            "sync_secs": object.__getattribute__(timer, "sync_secs"),
            "ops": {k: v for k, v in sorted(secs.items(), key=lambda kv: -kv[1])},
            "calls": calls,
        })
        ticks += 1
    object.__setattr__(timer, "on", False)
    total = time.perf_counter() - t_arm

    if req.phase == _PHASE_PREFILL:
        raise RuntimeError(f"prefill did not finish in {_MAX_TICKS} ticks")
    return {"tokens": n_tokens, "chunks": chunks, "prefill_secs": total,
            "n_chunks": len(chunks)}


def _bucket(ops: dict) -> dict:
    """Watched ops by name, everything else summed into `other` -- never dropped."""
    out = {k: v for k, v in ops.items() if k in _WATCH}
    other = sum(v for k, v in ops.items() if k not in _WATCH)
    if other:
        out["other"] = other
    return out


def _report(arm: dict) -> None:
    n, chunks = arm["tokens"], arm["chunks"]
    print(f"\n=== {n} tokens, {arm['n_chunks']} chunks, "
          f"{arm['prefill_secs']:.3f} s total ===")
    print(f"{'prefix':>7} {'chunk':>6} {'tick_s':>8}   top ops")
    for c in chunks:
        top = sorted(_bucket(c["ops"]).items(), key=lambda kv: -kv[1])[:4]
        s = "  ".join(f"{k} {v:.3f}" for k, v in top)
        print(f"{c['prefix']:>7} {c['chunk']:>6} {c['tick_secs']:>8.3f}   {s}")

    tot = defaultdict(float)
    for c in chunks:
        for k, v in _bucket(c["ops"]).items():
            tot[k] += v
    whole = sum(tot.values()) or 1.0
    print(f"\n{'op':<22}{'secs':>9}{'share':>8}{'calls':>9}")
    ncalls = defaultdict(int)
    for c in chunks:
        for k, v in c["calls"].items():
            ncalls[k if k in _WATCH else "other"] += v
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"{k:<22}{v:>9.3f}{v / whole * 100:>7.1f}%{ncalls[k]:>9}")
    print(f"{'(sync overhead)':<22}{sum(c['sync_secs'] for c in chunks):>9.3f}")

    # The discriminator: does an op's per-chunk cost rise with the prefix?
    if len(chunks) >= 3:
        print("\nper-chunk trend (first chunk -> last), the n^2 signature:")
        first, last = chunks[0], chunks[-1]
        fb, lb = _bucket(first["ops"]), _bucket(last["ops"])
        span = last["prefix"] / max(first["prefix"], 1) if first["prefix"] else float("inf")
        print(f"  prefix {first['prefix']} -> {last['prefix']}"
              + (f" ({span:.1f}x)" if span != float("inf") else ""))
        for k in sorted(set(fb) | set(lb), key=lambda k: -(lb.get(k, 0))):
            a, b = fb.get(k, 0.0), lb.get(k, 0.0)
            if max(a, b) < 1e-4:
                continue
            r = f"{b / a:.2f}x" if a > 0 else "n/a"
            print(f"    {k:<20} {a:>8.4f} -> {b:>8.4f}   {r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--tokens", default="512,1024,2048")
    ap.add_argument("--json")
    ap.add_argument("--selfcheck", action="store_true",
                    help="CPU gate: assert the instrument sees what it claims to")
    args = ap.parse_args()

    lengths = [int(x) for x in args.tokens.split(",")]
    backend = get_backend()
    prov = _provenance(getattr(backend, "precision", "bf16"))
    print(f"arch={backend.arch} target={backend.target} "
          f"precision={getattr(backend, 'precision', '?')}")
    mine = prov.get(backend.arch, {})
    for op in ("paged_attention", "paged_attention_split", "attn_prep", "gdn_chunk_fused"):
        here = mine.get(op)
        others = {a: t[op] for a, t in prov.items() if op in t and a != backend.arch}
        shared = [a for a, t in others.items() if t == here]
        print(f"  {op:<24} {here or '-- not registered':<40}"
              + (f"same maker as: {','.join(shared)}" if shared else ""))

    if args.model != "tiny":
        raise SystemExit("only the tiny/tiny-agent config is wired here; "
                         "pass --model tiny and set --tokens")

    ctx = max(lengths) + 64
    cfg = tiny(max_position_embeddings=ctx)
    from tilerl.model import build_random
    timer = _Timer(backend)
    model = build_random(cfg, seed=7)
    engine = build_engine(cfg, model, timer, num_blocks=(ctx // 16) * 4 + 64,
                          num_slots=4, max_batch=1, max_total_tokens=ctx,
                          prefix_store=NoPrefixStore())

    arms = []
    for i, n in enumerate(lengths):
        arm = _prefill_arm(engine, timer, n, cfg.vocab_size, seed=100 + i)
        _report(arm)
        arms.append(arm)

    if args.selfcheck:
        _selfcheck(arms)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"arch": backend.arch, "provenance": prov, "arms": arms}, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


def _selfcheck(arms: list) -> None:
    """The instrument can be wrong in ways that still print a table.

    Each assertion is one way it could lie: seeing nothing, seeing only some
    ticks, or bucketing so that the parts do not add up to the whole.
    """
    long = max(arms, key=lambda a: a["tokens"])
    assert long["n_chunks"] >= 2, (
        f"{long['tokens']} tokens ran in {long['n_chunks']} chunk(s); the per-chunk "
        "trend is the whole point and needs at least two")

    for a in arms:
        for c in a["chunks"]:
            assert c["ops"], f"a tick at prefix {c['prefix']} recorded no op at all"
            attributed = sum(_bucket(c["ops"]).values())
            # ops nest (paged_attention calls _kernel, not another backend op), so
            # the sum is <= the tick; a sum ABOVE it means double counting.
            assert attributed <= c["tick_secs"] * 1.5 + 1e-3, (
                f"ops sum {attributed:.4f} exceeds tick {c['tick_secs']:.4f} at "
                f"prefix {c['prefix']}: the proxy is counting a call twice")

    # the prefix really advances, or "cost rises with prefix" is unmeasurable
    pref = [c["prefix"] for c in long["chunks"]]
    assert pref == sorted(pref) and pref[-1] > pref[0], f"prefix did not advance: {pref}"

    # The first chunk of a fresh prompt attends over NOTHING, so its prefix is 0.
    # Reading prefill_from after the step instead of before shifts every x by one
    # chunk, which still prints a rising series and a plausible fit -- the whole
    # per-op trend would then be attributed to the wrong prefix length.
    assert pref[0] == 0, (
        f"first chunk recorded prefix {pref[0]}, not 0: prefill_from is being read "
        "after the step, so every chunk's prefix is one chunk too high")
    assert sum(c["chunk"] for c in long["chunks"]) == long["tokens"], (
        f"chunks sum to {sum(c['chunk'] for c in long['chunks'])}, not "
        f"{long['tokens']}: a tick was missed or double-counted")

    # attention must be visible; if it is not, the proxy is not on the call path
    seen = {k for a in arms for c in a["chunks"] for k in c["ops"]}
    assert seen & {"paged_attention", "attention"}, (
        f"no attention op was timed; the proxy missed the call path. saw: {sorted(seen)}")
    print(f"\nselfcheck OK: {len(arms)} arms, {long['n_chunks']} chunks at "
          f"{long['tokens']} tokens, prefix {pref[0]} -> {pref[-1]}, "
          f"{len(seen)} distinct ops timed")


if __name__ == "__main__":
    raise SystemExit(main())
