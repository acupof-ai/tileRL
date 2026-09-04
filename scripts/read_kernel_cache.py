"""Which shape axes does the tilelang cache say a kernel was specialized on?

Reads `params.pkl` from a tilelang kernel cache and reports, per kernel, the distinct
values each dimension took -- so "is B a shape axis" is answered from the artifact
instead of from the source. Used to settle #74 (B costs 41-42 compiles of 76-77) and
to confirm #73's fix left no post-fix entries.

Four things this exists to stop repeating:

1. `params.pkl` holds the FULL kernel signature -- declared inputs, scalars, AND the
   `T.empty` outputs. `paged_attention_split` declares 6 tensors and stores 10.
2. It holds no names, so a kernel is identified by its (rank, dtype) signature. Two
   kernels can share a rank tuple and differ only in one dtype: PO f16 is the live
   `paged_attention_split`, PO f32 is the pre-#44 one, 634 entries against 39.
3. Symbolic dims come back as tvm `Var`, not int. A dim that is still symbolic is not
   a shape axis; `int()` on it raises.
4. The cache is cumulative across every run of the tree, so a claim about the CURRENT
   code needs an mtime window (`--since`).

    python3 scripts/read_kernel_cache.py --cache ~/.tilelang/cache/0.1.13/linux-x86_64/kernels
    python3 scripts/read_kernel_cache.py --since 6c6f6df --repo ~/tilerl-git   # post-fix only
    python3 scripts/read_kernel_cache.py --sig 4f32,4f32,4f32,4f32,2i32,1i32,1i32  # one kernel
"""

from __future__ import annotations

import argparse
import collections
import datetime
import pathlib
import pickle
import subprocess

DT = {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "int32": "i32",
      "int64": "i64", "uint8": "u8", "float8_e4m3": "f8"}


def dims(x) -> tuple:
    """Shape with ints where concrete, the symbol's name where still symbolic."""
    out = []
    for v in getattr(x, "shape", ()):
        s = str(v)
        out.append(int(s) if s.lstrip("-").isdigit() else s)
    return tuple(out)


def sig(xs) -> str:
    """`4f32,2i32,...` -- rank and dtype per param, the only identity available."""
    return ",".join(
        f"{len(dims(x))}{DT.get(str(getattr(x, 'dtype', '')), str(getattr(x, 'dtype', '')))}"
        for x in xs
    )


def load(cache: pathlib.Path, since: float | None) -> dict[str, list]:
    by_sig: dict[str, list] = collections.defaultdict(list)
    for d in cache.iterdir():
        f = d / "params.pkl"
        if not f.is_file():
            continue
        t = f.stat().st_mtime
        if since is not None and t <= since:
            continue
        try:
            xs = pickle.loads(f.read_bytes())
        except Exception:
            continue
        xs = xs if isinstance(xs, (list, tuple)) else [xs]
        by_sig[sig(xs)].append((t, [dims(x) for x in xs]))
    return by_sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path,
                    default=pathlib.Path.home() / ".tilelang/cache/0.1.13/linux-x86_64/kernels")
    ap.add_argument("--sig", help="only this (rank, dtype) signature")
    ap.add_argument("--since", help="git rev: ignore entries written at or before its commit time")
    ap.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--top", type=int, default=6, help="how many signatures to detail")
    a = ap.parse_args()

    cut = None
    if a.since:
        out = subprocess.run(["git", "-C", str(a.repo), "log", "-1", "--format=%ct", a.since],
                             capture_output=True, text=True)
        cut = int(out.stdout.strip())
        print(f"# entries after {a.since} ({datetime.datetime.fromtimestamp(cut):%m-%d %H:%M})")

    by_sig = load(a.cache, cut)
    if not by_sig:
        print("no entries" + (" in that window" if cut else ""))
        return

    order = sorted(by_sig.items(), key=lambda kv: -len(kv[1]))
    for s, entries in order[: a.top] if not a.sig else [(a.sig, by_sig.get(a.sig, []))]:
        if not entries:
            print(f"\n{s}: no entries")
            continue
        ts = [t for t, _ in entries]
        print(f"\n{s}\n  {len(entries)} entries, "
              f"{datetime.datetime.fromtimestamp(min(ts)):%m-%d %H:%M}"
              f" .. {datetime.datetime.fromtimestamp(max(ts)):%m-%d %H:%M}")
        # Per param, per dimension: how many distinct values, i.e. is it an axis?
        nparam = len(entries[0][1])
        for p in range(nparam):
            shapes = [e[1][p] for e in entries if len(e[1]) > p]
            if not shapes or not shapes[0]:
                continue
            cols = []
            for i in range(len(shapes[0])):
                vals = {sh[i] for sh in shapes if len(sh) > i}
                cols.append(f"{len(vals)}" if len(vals) > 1 else f"={next(iter(vals))}")
            print(f"    p{p}: [{', '.join(cols)}]")
        # Distinct whole-shape combinations of param 0: the compile count that matters.
        p0 = collections.Counter(e[1][0] for e in entries)
        print(f"    p0 distinct shapes: {len(p0)}")
        if len(p0) <= 12:
            for sh, n in sorted(p0.items(), key=lambda kv: str(kv[0])):
                print(f"      {sh} x{n}")

    print(f"\n{len(by_sig)} signatures total; '=N' means that dim never varied "
          f"(not an axis), a bare count means it did.")


if __name__ == "__main__":
    main()
