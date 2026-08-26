"""Shared harness for pod kernel A/B and sweep runs (GPU 6,7).

Convention (docs/design-kernels.md, "SOTA iteration loop"): a new kernel
idea is a variant in a family script; one pod run A/Bs it against the
default and prints the bench-entry draft — paste the table into
docs/experience/. Family scripts build their own inputs at the canonical
shapes below and call ab(); benchkit is the process plumbing (timing,
relerr, table, entry draft).
"""

from __future__ import annotations

import os

import torch

# Canonical bench shapes (slice4 = 4 layers of the 27B, the workhorse).
SLICE4_SOURCE = "/host/tc27-nvfp4-slice4"
GDN_PREFILL = dict(B=1, T=512, QD=2048, nvh=48, K=128, V=128, KER=4)


def timeit(fn, iters=20, warmup=3):
    """Mean ms over `iters` GPU calls (cuda events), after `warmup` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def relerr(actual, ref):
    """Max abs error relative to ref's abs-max (0.0 on an all-zero ref)."""
    rmax = ref.abs().max().item()
    return (actual.float() - ref.float()).abs().max().item() / rmax if rmax > 0 else 0.0


def ab(name, arms, ref, iters=20, tol=1e-2):
    """A/B two+ arms against a reference tuple, same process, back-to-back.

    arms: list of (name, fn) where fn() -> tuple of tensors matching ref.
    Prints a markdown table with per-output relerr and the entry draft.
    Returns the rows (name, ms, errs, ok) for programmatic use.
    """
    rows = []
    for aname, fn in arms:
        outs = fn()
        torch.cuda.synchronize()
        errs = [relerr(o, r) for o, r in zip(outs, ref)]
        ms = timeit(fn, iters)
        rows.append((aname, ms, errs, max(errs) < tol))
    print(f"\n## {name}\n")
    print("| arm | ms | rel-err | verdict |")
    print("|---|---:|---:|---|")
    for aname, ms, errs, ok in rows:
        es = ", ".join(f"{e:.2e}" for e in errs)
        print(f"| {aname} | {ms:.4f} | {es} | {'OK' if ok else 'FAIL'} |")
    commit = os.environ.get("BENCH_COMMIT", "?")
    print(
        f"\nentry draft: H20 pod, cuda/sm90, commit {commit}, "
        f"mean of {iters} iters per arm, same process (contention-independent ratio)."
    )
    return rows
