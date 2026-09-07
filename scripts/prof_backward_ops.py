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
  per handler is overhead the shipped path does not pay (reported as its own figure).

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
    print(f"selfcheck ok: a drained handler timed {secs['slow']:.3f}s, an undrained call "
          f"{undrained * 1000:.3f}ms; nested {secs2['outer']:.3f}s outer + "
          f"{secs2['inner']:.3f}s inner, exclusive so the shares sum to 100%; "
          f"{len(pairs)} of main()'s imports resolved; --inside-gdn splits a stub into "
          f"{secs3['gdn_backward']:.3f}s remainder + {secs3['_gdn_chunk_fwd']:.3f}s recompute "
          f"+ {secs3['_gdn_chunk_bwd']:.3f}s adjoint, with the M solve "
          f"({secs3['solve_triangular (in recompute)'] * 1000:.3f}ms) taken OUT of recompute")
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

    secs, calls = instrument_gdn(sync) if a.inside_gdn else instrument(sync)
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
                          # T crosses this cap and the WY multiple independently, so two
                          # shapes can differ in both at once and did (T=1280 vs 1324)
                          "segment": "layer" if t_batch > _MLP_SEGMENT_MAX_T else "mlp"},
                         sort_keys=True), flush=True)

        secs.clear(); calls.clear()  # keep only the LAST step: step 0 pays every JIT
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        rl_step(model, batch, adv, plens, backend, optimizer, trainable=trainable,
                seq_lens=slens, micro=a.micro, timings=timings)
        sync()
        rows_out.append({"step": step + 1, "train_secs": round(time.perf_counter() - t0, 4),
                         "backward_secs": round(timings.get("backward_secs", 0.0), 4)})
        print(json.dumps(rows_out[-1], sort_keys=True), flush=True)

    attributed = sum(secs.values())
    bwd = rows_out[-1]["backward_secs"]
    table = _rows(secs, calls, attributed)
    print(f"\n# backward_secs {bwd:.3f} (rl_step's own timing, no per-handler sync)")
    print(f"# attributed to handlers {attributed:.3f} over {sum(calls.values())} calls")
    # A RESIDUAL, named as one. It was labelled "sync overhead this probe adds" and printed
    # -4.797 s: a negative overhead is a contradiction, and the sign says the handlers do not
    # account for all of backward_secs. What is outside them is Tape.backward's own loop --
    # grads dict arithmetic, the `grads[tid] + g_in` accumulate, _release, entry bookkeeping --
    # plus whatever the per-handler syncs add on top, which is why this is a net figure and
    # not a measurement of either term.
    resid = bwd - attributed
    print(f"# unattributed: {resid:+.3f} s ({resid / bwd * 100:+.1f}% of backward_secs) -- "
          f"Tape.backward's own loop and bookkeeping, NET of the per-handler sync overhead "
          f"this probe adds. Shares are within the attributed total; absolute seconds are not "
          f"the shipped path's")
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
