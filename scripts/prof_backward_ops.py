"""Where does the backward's 69.7 s go, op by op?

One GRPO step is 133 s at gen 1024 and `backward_secs` is 69.7 of it — 52%
(wins/2026-09-06-one-grpo-step-is-54-percent-backward.md). That number is one bucket:
`rl_step` times the whole `tape.backward` call and nothing splits it. The next lever should
come from a table, not from a guess about which op is slow.

This wraps `autograd._BWD` so every handler is timed by op name, with a device sync around
each call. Two things that makes it, and two it does not:

* per-op wall time inside ONE warm backward, summed by name, with a call count. A handler
  that runs 1024 times cheaply and one that runs twice expensively are different levers.
* the recomputed forwards separated from the gradient work: `checkpoint`'s handler replays
  its segment on a sub-tape, so its own line is recompute + the segment's own ops, and the
  sub-tape's ops are attributed to their own names by the same wrapper.
* NOT a kernel profile. A handler's time includes python, dispatch and allocation; a slow
  line says "look here", not "this kernel is slow".
* NOT comparable across trees or across box states — one process, one step, and the sync
  per handler is overhead the shipped path does not pay. `--no-instrument` is that arm:
  measured at C=128, 31.339 s bare against 34.717 (`--inside-gdn`) and 35.620 (registry).
  Each difference bounds ITS OWN arm's instrument; there is no per-call cost across arms,
  because the two wrappers sit at different depths and their call counts measure different
  things (fitting one gives -102.8 us/call — the arm with more timed calls is the faster).
* NOT a claim about what the unattributed remainder is. It is dominated by the ops the
  chosen arm does not wrap: `--inside-gdn` leaves 26.918 s at C=128, registry mode 4.831 s
  for the same work.

  TILERL_TARGET=cpu python3 scripts/prof_backward_ops.py --selfcheck   # no GPU
  scripts/pod_run.sh bwdops 6 -- python3 -u scripts/prof_backward_ops.py --gen 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "packages" / "tilerl-kernels" / "src")
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tilerl import autograd as ag  # noqa: E402

#: Which backward each op resolves to, and whether that is a TileLang kernel or torch-eager.
#: Read off backend.py, not guessed: rmsnorm_bwd calls rmsnorm_rstd + rmsnorm_bwd_x (:536),
#: linear_bwd calls gemm_nn (:605), linear_frozen_bwd calls linear_fp4_bwd when the scale
#: block is 16 (:1107). The rest carry `# ponytail: torch-eager backward`.
KIND = {
    "rmsnorm": "kernel: rmsnorm_rstd + rmsnorm_bwd_x",
    "rmsnorm_f32": "kernel: rmsnorm_rstd + rmsnorm_bwd_x",
    "linear": "kernel: gemm_nn",
    "linear_fp4_frozen": "kernel: linear_fp4_bwd (sm90)",
    #: the kernel is gated `not fp8` (:1118), so fp8 falls through to the reference at :1136
    "linear_fp8_frozen": "eager (no fp4_bwd for fp8)",
    "rope": "eager",
    "attention": "eager",
    "paged_attention": "eager",
    "linear_attn_chunk": "eager (GDN)",
    #: --inside-gdn rows
    "_gdn_chunk_fwd": "eager: the recompute, exclusive of the M solve",
    "_gdn_chunk_bwd": "eager: the adjoint",
    "gdn_backward": "eager: prologue + epilogue remainder",
    "solve_triangular (in recompute)": "torch/cuBLAS batched triangular solve",
    "silu_mul": "eager",
    "embedding": "eager",
    "reshape": "view",
    "slice": "view",
    "add": "view",
    "checkpoint": "recompute + the segment's own ops",
    "all_reduce": "collective",
    "all_gather": "collective",
    "cp_gather": "collective",
    "tp_fork": "collective",
}


def instrument(sync) -> tuple[dict, dict]:
    """Wrap every handler in `autograd._BWD` with a timer. Returns (secs, calls) by op name.

    Wrapping the REGISTRY rather than the loop: `Tape.backward` resolves `_BWD[op_name]` at
    dispatch, and a sub-tape from `checkpoint` resolves the same dict, so a segment's inner
    ops are attributed to their own names with no second hook.

    Time is EXCLUSIVE: `checkpoint`'s handler replays its segment through this same registry
    (measured on tiny: 12 nested calls), so a naive elapsed-time wrapper counts the inner
    `linear` and `rmsnorm` in both their own rows and in `checkpoint`'s, and the shares sum
    past 100%. Each frame subtracts what its callees took, so `checkpoint`'s row is the
    recompute forward alone and the rows are disjoint.
    """
    secs: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    child = [0.0]  # time attributed to callees of the frame currently running

    def wrap(name, fn):
        def timed(*a, **kw):
            sync()
            outer, child[0] = child[0], 0.0
            t0 = time.perf_counter()
            # A handler is a GENERATOR: calling it runs nothing. Draining it here is what
            # makes the time real -- `yield from` in the caller would leave every line 0.
            out = list(fn(*a, **kw))
            sync()
            elapsed = time.perf_counter() - t0
            secs[name] += elapsed - child[0]  # exclusive
            calls[name] += 1
            child[0] = outer + elapsed        # this frame is a callee of the one above
            return iter(out)

        timed.wants = getattr(fn, "wants", False)  # _BWD carries this flag on some handlers
        return timed

    for name, fn in list(ag._BWD.items()):
        ag._BWD[name] = wrap(name, fn)
    return secs, calls


def _rows(secs, calls, total):
    out = []
    for name in sorted(secs, key=lambda n: -secs[n]):
        if calls[name] == 0:
            continue
        out.append({
            "op": name,
            "secs": round(secs[name], 4),
            "share": round(secs[name] / total * 100, 2) if total else 0.0,
            "calls": calls[name],
            "ms_per_call": round(secs[name] / calls[name] * 1000, 3),
            "kind": KIND.get(name, "?"),
        })
    return out


def _selfcheck() -> int:
    """The timing wrapper on a scripted tape: no model, no GPU.

    The load-bearing property is that a GENERATOR handler is drained inside the timer. The
    obvious wrapper returns `fn(*a)` and every row reads 0.0 s while the work happens in the
    caller -- a table of zeros that looks like a fast backward.
    """
    slept = {"n": 0}

    def slow(backend, g, args, kw):
        slept["n"] += 1
        time.sleep(0.02)
        yield 0, g

    saved = dict(ag._BWD)
    try:
        ag._BWD.clear()
        ag._BWD["slow"] = slow
        secs, calls = instrument(lambda: None)
        list(ag._BWD["slow"](None, torch.zeros(1), (torch.zeros(1),), {}))
        assert calls["slow"] == 1, calls
        assert secs["slow"] >= 0.015, f"the generator was not drained inside the timer: {secs}"
        # and the control: a wrapper that does not drain reads ~0
        t0 = time.perf_counter()
        saved_gen = slow(None, torch.zeros(1), (torch.zeros(1),), {})
        undrained = time.perf_counter() - t0
        assert undrained < 0.005, f"expected an undrained call to be ~free, got {undrained}"
        list(saved_gen)

        # Nesting: a handler that dispatches through the registry (which is what
        # `checkpoint` does -- 12 nested calls on tiny) must not be charged for its callees.
        # An inclusive timer put 24.20 ms on checkpoint against 2.10 exclusive, and the
        # shares summed past 100%.
        ag._BWD.clear()

        def inner(backend, g, args, kw):
            time.sleep(0.02)
            yield 0, g

        def outer(backend, g, args, kw):
            time.sleep(0.02)
            list(ag._BWD["inner"](backend, g, args, kw))
            yield 0, g

        ag._BWD["inner"], ag._BWD["outer"] = inner, outer
        secs2, calls2 = instrument(lambda: None)
        list(ag._BWD["outer"](None, torch.zeros(1), (torch.zeros(1),), {}))
        assert calls2 == {"outer": 1, "inner": 1}, calls2
        assert 0.015 <= secs2["inner"] < 0.035, f"inner should be its own sleep: {secs2}"
        assert 0.015 <= secs2["outer"] < 0.035, (
            f"outer is charged for inner -- the timer is inclusive: {dict(secs2)}")
        total, elapsed = sum(secs2.values()), secs2["outer"] + secs2["inner"]
        assert abs(total - elapsed) < 1e-9 and total >= 0.03, (total, elapsed)

        # main()'s imports, resolved. --selfcheck returns before reaching them, so four
        # invented module paths (tilerl.lora / .optim / .sampling) survived every local run
        # and died 40 s into a pod run with ModuleNotFoundError. Importing them here is the
        # cheapest gate that fails on this machine instead of on the card.
        import importlib
        pairs = (("tilerl.model", "add_lora"), ("tilerl.autograd", "AdamW"),
                 ("tilerl.engine", "SamplingParams"), ("tilerl.engine", "build_engine"),
                 ("tilerl.cli", "_build_model"), ("tilerl.kv_cache", "NoPrefixStore"),
                 ("tilerl.train", "rl_step"), ("tilerl.train", "group_advantages"),
                 ("tilerl.train", "untruncated"))
        for mod, name in pairs:
            assert hasattr(importlib.import_module(mod), name), f"{mod} has no {name}"

        # --inside-gdn: the three names it patches must exist on the reference module, and
        # gdn_backward must reach the two helpers through module globals or the patch sees
        # nothing. A stub gdn_backward that calls them proves the wiring; the real one is
        # what the pod run exercises.
        ref = importlib.import_module("tilerl_kernels.reference")
        for name in ("_gdn_chunk_fwd", "_gdn_chunk_bwd", "gdn_backward"):
            assert callable(getattr(ref, name, None)), f"reference has no {name}"
        real = {n: getattr(ref, n) for n in ("_gdn_chunk_fwd", "_gdn_chunk_bwd",
                                             "gdn_backward")}
        real_solve = torch.linalg.solve_triangular
        try:
            ref._gdn_chunk_fwd = lambda *a: (time.sleep(0.01), _tiny_solve())[0]
            ref._gdn_chunk_bwd = lambda *a: time.sleep(0.02)
            # calls the helpers by GLOBAL name, as gdn_backward:929/:955 do
            def stub(*a, **kw):
                time.sleep(0.02)          # stands in for the prologue/epilogue
                ref._gdn_chunk_fwd()
                ref._gdn_chunk_bwd()
            ref.gdn_backward = stub
            secs3, calls3 = instrument_gdn(lambda: None)
            ref.gdn_backward()
            assert calls3 == {"gdn_backward": 1, "_gdn_chunk_fwd": 1, "_gdn_chunk_bwd": 1,
                              "solve_triangular (in recompute)": 1}, dict(calls3)
            for n in ("_gdn_chunk_fwd", "_gdn_chunk_bwd"):
                assert 0.008 <= secs3[n] < 0.035, f"{n} should be its own sleep: {dict(secs3)}"
            assert 0.015 <= secs3["gdn_backward"] < 0.035, (
                "gdn_backward is charged for the helpers -- the timer is inclusive, and its "
                f"row is meant to be the prologue/epilogue remainder: {dict(secs3)}")
            # the solve nests inside the recompute, so its time comes OUT of that row
            solve = secs3["solve_triangular (in recompute)"]
            assert solve > 0, "the solve row is empty -- torch.linalg was not patched"
            assert secs3["_gdn_chunk_fwd"] + solve >= 0.01, (
                f"recompute + solve should cover the fwd stub: {dict(secs3)}")
        finally:
            for n, fn in real.items():
                setattr(ref, n, fn)
            torch.linalg.solve_triangular = real_solve   # a global patch, unlike the others
    finally:
        ag._BWD.clear()
        ag._BWD.update(saved)

    # --frozen-shapes: fp4's wq is [N, K/2] packed and fp8's is [N, K], so one wrapper reading
    # both must double only the fp4 width. A recorder that gets K wrong halves or doubles every
    # FLOP figure derived from it, and nothing downstream would look wrong.
    saved = dict(ag._BWD)
    try:
        ag._BWD.clear()
        for nm in ("linear_fp4_frozen", "linear_fp8_frozen"):
            ag._BWD[nm] = lambda backend, g, args, kw: iter(((0, g),))
        shapes = instrument_shapes()
        g4 = torch.zeros(2, 3, 8)                     # M = 6 after the reshape, N = 8
        list(ag._BWD["linear_fp4_frozen"](None, g4, (None, torch.zeros(8, 5)), {}))
        list(ag._BWD["linear_fp8_frozen"](None, g4, (None, torch.zeros(8, 5)), {}))
        assert shapes[("linear_fp4_frozen", 6, 8, 10)] == 1, dict(shapes)  # 5 * 2 = 10
        assert shapes[("linear_fp8_frozen", 6, 8, 5)] == 1, dict(shapes)   # 5 as-is
        assert len(shapes) == 2, dict(shapes)
    finally:
        ag._BWD.clear()
        ag._BWD.update(saved)
    print(f"selfcheck ok: a drained handler timed {secs['slow']:.3f}s, an undrained call "
          f"{undrained * 1000:.3f}ms; nested {secs2['outer']:.3f}s outer + "
          f"{secs2['inner']:.3f}s inner, exclusive so the shares sum to 100%; "
          f"{len(pairs)} of main()'s imports resolved; --inside-gdn splits a stub into "
          f"{secs3['gdn_backward']:.3f}s remainder + {secs3['_gdn_chunk_fwd']:.3f}s recompute "
          f"+ {secs3['_gdn_chunk_bwd']:.3f}s adjoint, with the M solve "
          f"({secs3['solve_triangular (in recompute)'] * 1000:.3f}ms) taken OUT of recompute; "
          f"the shape recorder doubles fp4's packed K (5 -> 10) and leaves fp8's at 5")
    return 0


def _tiny_solve():
    """A real solve_triangular call, so the selfcheck's recompute stub reaches the patched
    `torch.linalg` the way `_gdn_chunk_fwd:611` does."""
    n = 4
    eye = torch.eye(n)
    return torch.linalg.solve_triangular(eye, eye, upper=False, unitriangular=True)


def instrument_gdn(sync) -> tuple[dict, dict]:
    """Split `reference.gdn_backward` — the 45.3 s row — into its own sub-calls.

    Two functions carry almost all of it and both are module-level, so patching
    `reference._gdn_chunk_fwd` / `_gdn_chunk_bwd` catches every call:

    * `_gdn_chunk_fwd` is called from `gdn_backward:929` to RECOMPUTE the forward chunk
      loop, because the tape keeps no chunk intermediates. That time is recompute, not
      gradient work, and no upstream backward kernel replaces it — a tape change would.
    * `_gdn_chunk_bwd` (:955) is the adjoint proper, the part `example_chunk_delta_bwd`
      and `example_wy_fast_bwd_split` would replace.

    `gdn_backward`'s own prologue (conv1d taps, silu, the two L2 norms, softplus) and
    epilogue (norm/silu/conv adjoints, the head-group folds) are straight-line code, not
    functions, so they are not wrapped. They fall out as `gdn_backward` minus the two
    helpers, and that remainder is reported as its own row rather than left implicit.

    One more row nests below the recompute: the `M` solve (`reference.py:611`) is reached as
    `torch.linalg.solve_triangular`, so it is patched on `torch.linalg` — a PROCESS-WIDE
    mutation, unlike the others, and not restored (this probe runs one profile and exits).
    Its time comes out of `_gdn_chunk_fwd`'s row, not in addition to it.
    """
    secs: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    child = [0.0]

    def wrap(name, fn):
        def timed(*a, **kw):
            sync()
            outer, child[0] = child[0], 0.0
            t0 = time.perf_counter()
            out = fn(*a, **kw)  # NOT a generator, unlike a _BWD handler: no drain needed
            sync()
            elapsed = time.perf_counter() - t0
            secs[name] += elapsed - child[0]
            calls[name] += 1
            child[0] = outer + elapsed
            return out

        return timed

    from tilerl_kernels import reference as ref
    # gdn_backward wrapped too, so the prologue/epilogue remainder is a measured
    # subtraction rather than an assumption about what is left over.
    for name in ("_gdn_chunk_fwd", "_gdn_chunk_bwd", "gdn_backward"):
        setattr(ref, name, wrap(name, getattr(ref, name)))
    # The M solve (reference.py:611) is reached as `torch.linalg.solve_triangular`, an
    # attribute of torch.linalg rather than a global of `reference`, so patching `ref`
    # would miss it. It nests inside _gdn_chunk_fwd, and the exclusive timer already
    # subtracts a callee from its caller, so this row comes OUT of the recompute row.
    torch.linalg.solve_triangular = wrap(
        "solve_triangular (in recompute)", torch.linalg.solve_triangular)
    return secs, calls


def instrument_shapes() -> dict:
    """Count frozen-linear backward calls by (op, M, N, K). Returns the counter.

    `_frozen`'s handler (autograd.py:247-252) gets `g` [.., N] and `args[1]` = wq, whose shape
    is [N, K/2] for fp4 (packed nibbles) and [N, K] for fp8. M is g's flattened leading extent
    -- the same reshape `linear_frozen_bwd` does, so M is the row count the GEMM actually sees
    rather than the step's token total.

    This exists because arithmetic did not reconcile: 8 micro-batches x 56 fp4 layers x 2 fused
    linears x 2 (checkpoint replays `_mlp_body`) predicts 1792 calls and the measured count is
    2112. Factoring 2112 produces several decompositions that fit and none that is evidence, so
    the call sites are asked directly.
    """
    shapes: dict = defaultdict(int)

    def wrap(name, fn, fp8):
        def handler(backend, g, args, kw):
            wq = args[1]
            n = wq.shape[0]
            k = wq.shape[1] if fp8 else wq.shape[1] * 2
            shapes[(name, int(g.reshape(-1, g.shape[-1]).shape[0]), int(n), int(k))] += 1
            yield from fn(backend, g, args, kw)

        return handler

    for name, fp8 in (("linear_fp4_frozen", False), ("linear_fp8_frozen", True)):
        ag._BWD[name] = wrap(name, ag._BWD[name], fp8)
    return shapes


def bench_dx_gemms(shapes: dict, sync, reps: int = 12) -> list[dict]:
    """Time the dX contraction each frozen row runs -- g [M,N] @ W [N,K] -> [M,K], bf16.

    Not a square GEMM at "the shape": dX is what `linear_frozen_bwd` computes, so a different
    contraction would floor the wrong quantity. Reports the spread as well as the median,
    because 13.046 s / (one achieved TFLOP/s) inherits that width.

    CONDITION, reported with the number: this runs after a 27B backward in the same process, so
    the model is resident (~64 GiB) and the clocks have been under sustained load. That makes
    it the right place to learn the shapes and a possibly pessimistic place to read peak
    throughput -- a low floor inflates any gap measured against it, which is the direction to
    distrust.
    """
    out = []
    for (op, m, n, k), count in sorted(shapes.items(), key=lambda kv: -kv[1] * kv[0][2] * kv[0][3]):
        g = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            g @ w
        sync()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            g @ w
            sync()
            times.append(time.perf_counter() - t0)
        times.sort()
        flop = 2 * m * n * k
        med = times[len(times) // 2]
        out.append({
            "op": op, "M": m, "N": n, "K": k, "calls": count,
            "median_ms": round(med * 1e3, 4),
            "min_ms": round(times[0] * 1e3, 4),
            "max_ms": round(times[-1] * 1e3, 4),
            "tflops_median": round(flop / med / 1e12, 2),
            "tflops_min": round(flop / times[-1] / 1e12, 2),
            "tflops_max": round(flop / times[0] / 1e12, 2),
            "row_flop_tf": round(flop * count / 1e12, 2),
        })
        del g, w
        torch.cuda.empty_cache()
    return out


def bench_fp4_bwd(backend, shapes: dict, sync, reps: int = 12) -> list[dict]:
    """Time `linear_frozen_bwd` BARE at each fp4 shape, beside the bf16 GEMM at the same shape.

    This is the denominator any tile claim needs. Without it the row's 13.046 s carries the
    per-handler timer's hooks (~10% on the arms where that was measured), so a ratio against a
    GEMM floor mixes kernel time with instrument cost. Calling the backend method directly pays
    neither the timer nor the tape.

    fp4 only: the fp8 path has no kernel (`linear_frozen_bwd` gates on `not fp8`), so timing it
    here would re-measure the eager reference the row already reports.
    """
    out = []
    for (op, m, n, k), count in sorted(shapes.items(), key=lambda kv: -kv[1] * kv[0][2] * kv[0][3]):
        if op != "linear_fp4_frozen" or n < 1024:  # the N=48 rows are 21 ms total; skip
            continue
        g = torch.randn(m, n, dtype=torch.bfloat16, device=backend.device)
        # the shipped layout: packed nibbles [N, K/2] with an f32 scale per 16 columns
        wq = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device=backend.device)
        scale = torch.rand(n, k // 16, dtype=torch.float32, device=backend.device) * 0.01
        try:
            for _ in range(3):
                backend.linear_frozen_bwd(g, wq, scale)
            sync()
        except Exception as exc:  # a shape the kernel refuses is a finding, not a crash
            out.append({"op": op, "M": m, "N": n, "K": k, "error": repr(exc)[:200]})
            del g, wq, scale
            continue
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            backend.linear_frozen_bwd(g, wq, scale)
            sync()
            times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        flop = 2 * m * n * k
        out.append({
            "op": op, "M": m, "N": n, "K": k, "calls": count,
            "kernel_median_ms": round(med * 1e3, 4),
            "kernel_min_ms": round(times[0] * 1e3, 4),
            "kernel_max_ms": round(times[-1] * 1e3, 4),
            "kernel_tflops": round(flop / med / 1e12, 2),
            "row_secs_from_kernel": round(med * count, 4),
        })
        del g, wq, scale
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--inside-gdn", action="store_true",
                    help="split reference.gdn_backward instead of the _BWD registry")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--gen", type=int, default=1024)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2, help="step 0 pays the JIT; the last is warm")
    #: `_GDN_CHUNK` is read as a module global at call time (reference.py:923, :928), so
    #: setting it here reaches gdn_backward without touching the shipped default.
    ap.add_argument("--gdn-chunk", type=int, default=0,
                    help="override reference._GDN_CHUNK for this run (0 = the shipped value)")
    ap.add_argument("--count-kernels", action="store_true",
                    help="CUDA launch count for the warm step instead of per-op times; the "
                         "profiler distorts wall time, so never read seconds off this run")
    #: The control for what the per-op tables cannot see. BOTH instrument() and
    #: instrument_gdn() sync twice per handler call, so registry mode is not a sync-free arm --
    #: it only has fewer timed calls. Only this one leaves the shipped path alone, which makes
    #: each arm's attributed total an upper bound rather than a measurement.
    ap.add_argument("--no-instrument", action="store_true",
                    help="time the step with NO per-op hooks: backward_secs only, the "
                         "shipped path's own number")
    #: names the frozen-linear call sites and then floors them, in one process: the GEMM bench
    #: needs the shapes the count table reveals, so a separate run would guess them
    ap.add_argument("--frozen-shapes", action="store_true",
                    help="count frozen-linear backward calls by (op, M, N, K), then time the "
                         "bf16 dX GEMM at each shape; implies no per-op timing table")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selfcheck:
        return _selfcheck()

    # Copied from prof_grpo_step.py:47-54, not guessed: add_lora lives in model, AdamW in
    # autograd, SamplingParams in engine. My first version invented tilerl.lora/optim/sampling
    # and the run died 40 s in on the pod with ModuleNotFoundError.
    from tilerl_kernels.backend import get_backend

    from tilerl.autograd import AdamW
    from tilerl.cli import _build_model
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import add_lora
    from tilerl.train import group_advantages, rl_step, untruncated

    backend = get_backend()
    cuda = backend.device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    cfg, model = _build_model(a.model, seed=0, keep_master=False)
    engine = build_engine(cfg, model, backend, num_blocks=a.blocks, num_slots=a.group,
                          max_batch=a.group, max_total_tokens=a.blocks * 16,
                          decode_graph=False, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=a.rank)
    optimizer = AdamW(lr=1e-5)
    sampling = untruncated(SamplingParams(max_new_tokens=a.gen))
    rng = np.random.default_rng(0)
    vocab = int(getattr(cfg, "vocab_size", 0)) or 1000
    prompt = rng.integers(1, vocab, size=a.prompt_tokens, dtype=np.int64)

    shapes = None
    if a.frozen_shapes:
        secs, calls = defaultdict(float), defaultdict(int)
        shapes = instrument_shapes()
    elif a.no_instrument:
        secs, calls = defaultdict(float), defaultdict(int)
    else:
        secs, calls = instrument_gdn(sync) if a.inside_gdn else instrument(sync)
    from tilerl_kernels import reference
    if a.gdn_chunk:
        reference._GDN_CHUNK = a.gdn_chunk
    rows_out = []
    for step in range(a.steps):
        ids = [engine.submit(list(prompt), sampling) for _ in range(a.group)]
        comps: dict[int, list[int]] = {}
        while len(comps) < a.group:
            engine.step()
            comps.update(engine.poll())
        seqs = [comps[i] for i in ids]
        batch = np.stack([
            np.concatenate([prompt, np.asarray(c, dtype=np.int64),
                            np.zeros(a.gen - len(c), dtype=np.int64)]) for c in seqs
        ])
        adv = group_advantages(np.ones(a.group), a.group)
        plens = np.full(a.group, len(prompt), dtype=np.int64)
        slens = np.array([len(prompt) + len(c) for c in seqs], dtype=np.int64)
        # which GDN arm this shape takes, asked of the backend rather than recomputed here.
        # The t % chunk term binds only where gdn_state_scan is registered (sm90); on cpu the
        # predicate is vacuously true, so `gdn_arm` discriminates on the pod and not locally.
        from tilerl_kernels.backend import _WY_CHUNK
        from tilerl_kernels.registry import _resolve

        from tilerl.train import _MLP_SEGMENT_MAX_T
        t_batch = int(batch.shape[1])
        kset = _resolve(backend.precision, backend.arch)
        has_wy = "gdn_state_scan" in kset
        wy = backend._wy_eligible(t_batch, {"seq_q_lens": None}, t_batch > 1)
        # the same order linear_attn_chunk falls through: WY, then the fused kernel, then the
        # per-step reference. "not WY" is three different implementations, not one.
        if not has_wy:
            arm = "n/a (no WY cell on this arch)"
        elif wy:
            arm = "wy_kernels"
        elif t_batch > 1 and "gdn_chunk_fused" in kset:
            arm = "gdn_chunk_fused"
        else:
            arm = "reference.gdn_forward"
        print(json.dumps({"T": t_batch, "wy_chunk": _WY_CHUNK,
                          "T_mod_chunk": t_batch % _WY_CHUNK,
                          "wy_kernels_registered": has_wy,
                          "gdn_forward_arm": arm,
                          # read back from the module, not from the flag: the constant the
                          # backward actually uses is the only one worth recording
                          "gdn_chunk": reference._GDN_CHUNK,
                          # T crosses this cap and the WY multiple independently, so two
                          # shapes can differ in both at once and did (T=1280 vs 1324)
                          "segment": "layer" if t_batch > _MLP_SEGMENT_MAX_T else "mlp"},
                         sort_keys=True), flush=True)

        secs.clear(); calls.clear()  # keep only the LAST step: step 0 pays every JIT
        timings: dict[str, float] = {}
        warm = step == a.steps - 1
        t0 = time.perf_counter()
        if a.count_kernels and warm and cuda:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                        seq_lens=slens, micro=a.micro, timings=timings)
                sync()
            launches = sum(1 for e in prof.events() if e.device_type.name == "CUDA")
        else:
            rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                    seq_lens=slens, micro=a.micro, timings=timings)
            sync()
            launches = None
        row = {"step": step + 1, "train_secs": round(time.perf_counter() - t0, 4),
               "backward_secs": round(timings.get("backward_secs", 0.0), 4)}
        if cuda:
            # per-arm, because a cross-arm comparison of two arms at very different
            # footprints cannot rule out memory pressure without it
            row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            torch.cuda.reset_peak_memory_stats()
        if launches is not None:
            # under the profiler, so train_secs on this row is inflated and not comparable
            row["cuda_launches"] = launches
            row["profiled"] = True
        rows_out.append(row)
        print(json.dumps(rows_out[-1], sort_keys=True), flush=True)

    attributed = sum(secs.values())
    bwd = rows_out[-1]["backward_secs"]
    table = _rows(secs, calls, attributed)
    if a.frozen_shapes:
        total = sum(shapes.values())
        print(f"\n# backward_secs {bwd:.3f} (shape counting only, no per-op timers)")
        print(f"# {total} frozen-linear backward calls over {len(shapes)} distinct shapes")
        print(f"\n# {'op':<20} {'M':>7} {'N':>7} {'K':>7} {'calls':>7}")
        for (op, m, n, k), c in sorted(shapes.items(), key=lambda kv: (kv[0][0], -kv[1])):
            print(f"  {op:<20} {m:7d} {n:7d} {k:7d} {c:7d}")
        gemms = bench_dx_gemms(shapes, sync) if cuda else []
        if gemms:
            print("\n# bf16 dX GEMM (g[M,N] @ W[N,K]), 12 reps warm, MODEL RESIDENT so the")
            print("# clocks are post-backward -- a floor read here is possibly pessimistic")
            print(f"# {'op':<20} {'M':>7} {'N':>7} {'K':>7} {'med ms':>8} "
                  f"{'TFLOP/s':>8} {'min':>7} {'max':>7} {'row TF':>9}")
            for r in gemms:
                print(f"  {r['op']:<20} {r['M']:7d} {r['N']:7d} {r['K']:7d} "
                      f"{r['median_ms']:8.3f} {r['tflops_median']:8.2f} "
                      f"{r['tflops_min']:7.2f} {r['tflops_max']:7.2f} {r['row_flop_tf']:9.2f}")
        kern = bench_fp4_bwd(backend, shapes, sync) if cuda else []
        if kern:
            print("\n# linear_frozen_bwd BARE at the fp4 shapes, no tape and no timer -- the")
            print("# denominator for a tile claim. `bf16 ms` repeats the GEMM above for contrast;")
            print("# the kernel dequantizes inside the same launch, so it cannot match it.")
            print(f"# {'M':>7} {'N':>7} {'K':>7} {'kern ms':>8} {'TFLOP/s':>8} {'bf16 ms':>8} "
                  f"{'vs bf16':>8} {'row s':>8}")
            bf = {(r["M"], r["N"], r["K"]): r for r in gemms}
            for r in kern:
                if "error" in r:
                    print(f"  {r['M']:7d} {r['N']:7d} {r['K']:7d}  REFUSED: {r['error']}")
                    continue
                b = bf.get((r["M"], r["N"], r["K"]), {}).get("median_ms")
                ratio = f"{r['kernel_median_ms'] / b:8.2f}" if b else "       -"
                print(f"  {r['M']:7d} {r['N']:7d} {r['K']:7d} {r['kernel_median_ms']:8.3f} "
                      f"{r['kernel_tflops']:8.2f} {b if b else 0:8.3f} {ratio} "
                      f"{r['row_secs_from_kernel']:8.3f}")
            done = sum(r.get("row_secs_from_kernel", 0.0) for r in kern)
            print(f"# those shapes' rows from bare kernel time: {done:.3f} s "
                  f"(the timed arm reported 13.046 s for all three fp4 shapes)")
        if a.out:
            Path(a.out).write_text(json.dumps(
                {"rows": rows_out, "backward_secs": bwd,
                 "shapes": [{"op": o, "M": m, "N": n, "K": k, "calls": c}
                            for (o, m, n, k), c in sorted(shapes.items())],
                 "gemms": gemms, "fp4_kernel": kern}, indent=2, sort_keys=True))
        return 0
    if a.no_instrument:
        # The control: no hooks, so backward_secs is the shipped path's own number and the
        # difference against an instrumented run at the same chunk is the probe's sync cost.
        print(f"\n# backward_secs {bwd:.3f} -- NO instrumentation, the shipped path")
        print("# no per-op table: nothing was timed. Against the SAME arm's instrumented run "
              "this bounds that arm's instrument cost -- per arm only: the registry and "
              "--inside-gdn wrappers sit at different depths, so their call counts measure "
              "different things and no per-call cost is recoverable across them (fitting one "
              "gives -102.8 us/call, since the arm with more timed calls is the faster).")
        if a.out:
            Path(a.out).write_text(json.dumps(
                {"rows": rows_out, "backward_secs": bwd, "instrumented": False},
                indent=2, sort_keys=True))
        return 0
    # `rl_step` times the whole tape.backward call, so this figure CONTAINS the per-handler
    # syncs -- it is not a sync-free reading. --no-instrument is the arm that is.
    print(f"\n# backward_secs {bwd:.3f} (rl_step's own timing, syncs included)")
    print(f"# attributed to handlers {attributed:.3f} over {sum(calls.values())} calls")
    # A RESIDUAL, named as one. It was labelled "sync overhead this probe adds" and printed
    # -4.797 s: a negative overhead is a contradiction. What is outside the handlers is
    # dominated by the ops THIS arm does not wrap -- measured: --inside-gdn times three
    # reference functions and leaves 26.918 s at C=128, where the registry arm times every
    # op and leaves 4.831 s for the same work (backward_secs 34.717 vs 35.620, within 2.6%).
    # Tape.backward's own loop is inside it too but was never shown to dominate it.
    resid = bwd - attributed
    print(f"# unattributed: {resid:+.3f} s ({resid / bwd * 100:+.1f}% of backward_secs) -- "
          f"mostly the ops this arm does not wrap, plus Tape.backward's own loop and the "
          f"per-handler syncs. Shares are within the attributed total; absolute seconds are "
          f"not the shipped path's -- --no-instrument is that arm")
    print(f"\n# {'op':<20} {'secs':>9} {'share':>7} {'calls':>7} {'ms/call':>9}  kind")
    for r in table:
        print(f"  {r['op']:<20} {r['secs']:9.3f} {r['share']:6.2f}% {r['calls']:7d} "
              f"{r['ms_per_call']:9.3f}  {r['kind']}")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"rows": rows_out, "table": table, "backward_secs": bwd,
             "attributed_secs": round(attributed, 4)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
