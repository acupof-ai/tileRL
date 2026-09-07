"""Row 51: which sm90 kernels are on Hopper's wgmma, and which are still on Ampere's mma.sync.

`linear_fp4_bwd` ran the Ampere `mma.sync` on a Hopper card because `_THREADS=64` gave it a
2-warp consumer where wgmma issues from 4 (wins/2026-09-07-fp4-backward-warpgroup.md): 1.35x of
the whole GRPO backward from one call site. That was found by accident. This asks the question
for every kernel at once, from the cache, with no card and no fixes.

Three columns per kernel, because two of them alone mislead:

  consumer width   The correlate. `launch_bounds` is NOT: 192 appears on both sides of the wgmma
                   line (a 128-producer + 64-consumer split reads 192 and emits mma.sync, while
                   `linear_fp8` and `gemm_tn` also read 192 and emit wgmma). Reporting
                   launch_bounds alone is what made the first read of this wrong.
  file counts      One kernel name compiles many times (per shape, per tile). A name can be
                   wgmma at one shape and mma.sync at another, so the split is the finding.
  call site        Which `self._kernel(...)` line launches it and what it passes for threads.
                   Without this a mma.sync row is not actionable -- the fix is a literal at a
                   call site, and some kernels have several.

Read-only. Prints what is, proposes nothing.

  python3 scripts/audit_mma_class.py                     # on the pod, reads TILELANG_CACHE_DIR
  python3 scripts/audit_mma_class.py --cache /path       # elsewhere
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND = _ROOT / "packages/tilerl-kernels/src/tilerl_kernels/backend.py"
_REGISTRY = _ROOT / "packages/tilerl-kernels/src/tilerl_kernels/registry.py"

#: `extern "C" __global__ void <name>_kernel(`, the one line that names what was compiled.
_KERNEL_NAME = re.compile(r'__global__ void (?:__launch_bounds__\([^)]*\)\s*)?(\w+?)_kernel\(')
_LAUNCH = re.compile(r"__launch_bounds__\((\d+)")
#: the producer branch guard: `if (((int)threadIdx.x) < N)` splits producer from consumer.
_GUARD = re.compile(r"threadIdx\.x\)\s*<\s*(\d+)\)")


def _classify(src: str) -> dict | None:
    m = _KERNEL_NAME.search(src)
    if not m:
        return None
    lb = _LAUNCH.search(src)
    guard = _GUARD.search(src)
    total = int(lb.group(1)) if lb else None
    # No producer guard means the kernel is not warp-specialized: every thread is the consumer.
    consumer = (total - int(guard.group(1))) if (total and guard) else total
    # Three classes, not two. "no wgmma" is NOT "mma.sync": most kernels here (scatter, rope,
    # rmsnorm, silu) issue no matrix instruction at all, and folding them together reported
    # `write_tokens` as 1715 mma.sync cells -- naming a tensor-core problem in a kernel that
    # never touches a tensor core.
    cls = "wgmma" if "wgmma" in src else ("mma.sync" if "mma_sync" in src else "no MMA")
    return {"name": m.group(1), "cls": cls, "launch_bounds": total, "consumer": consumer}


def _call_sites() -> dict[str, list[str]]:
    """kernel name -> ["backend.py:LINE passes <threads expr>", ...].

    Reads the source rather than importing: the threads argument is the last positional of the
    launch, and several sites pass a literal or a local rather than `_THREADS`.

    A launch through a VARIABLE has to be followed or the table lies about shipped kernels --
    `linear_fp8` and the fp4->e4m3 arms are never named at their call site. `self._kernel(kernel)`
    takes its name from `_plan(op, ...)`, so the enclosing `def` names the op and `_CUDA_PLAN`'s
    rows for that op name the reachable kernels. Attributing every plan kernel to every such site
    instead (the first version of this) credited `linear_fp4_gemv` to an rmsnorm launch.
    `self._kernel(key)` is the rmsnorm family, whose names are built inline and are not plan rows.
    """
    text = _BACKEND.read_text()
    src = text.splitlines()
    out: dict[str, list[str]] = defaultdict(list)
    # ("op", "regime"): ("kernel", ...) -> op -> {kernel}
    by_op: dict[str, set[str]] = defaultdict(set)
    for op, kern in re.findall(r'\("(\w+)",\s*"\w+"\):\s*\("([^"]+)"', text):
        by_op[op].add(kern)
    for i, line in enumerate(src, 1):
        m = re.search(r'self\._kernel\(\s*(?:"([^"]+)"|(\w+))', line)
        if not m:
            continue
        # the launch's arguments may wrap over several lines; take up to the closing paren
        blob, depth = "", 0
        for ln in src[i - 1:i + 12]:
            blob += ln
            depth += ln.count("(") - ln.count(")")
            if depth <= 0 and blob.count("(") > 1:
                break
        args = blob.split(")(")[-1] if ")(" in blob else blob
        thr = next((t for t in ("_THREADS", "thr", "128", "64", "32")
                    if re.search(rf"\b{re.escape(t)}\b", args)), "no threads arg")
        if m.group(1):
            out[m.group(1)].append(f"backend.py:{i} passes {thr}")
            continue
        if m.group(2) != "kernel":  # `key`: rmsnorm, not a plan row
            continue
        # the enclosing def names the op whose plan rows this site can launch
        enclosing = next((ln for ln in reversed(src[:i]) if re.match(r"    def \w+", ln)), "")
        op = re.match(r"    def (\w+)", enclosing).group(1) if enclosing else ""
        op = op.removesuffix("_bwd") if op.endswith("_bwd") else op
        for name in sorted(by_op.get(op, ())):
            out[name].append(f"backend.py:{i} (_CUDA_PLAN via {op}) passes {thr}")
    return out


def _registry_names() -> dict[str, str]:
    """registry key -> factory name, so an aliased kernel can be traced back to its call site."""
    out = {}
    for line in _REGISTRY.read_text().splitlines():
        m = re.search(r'"([^"]+)":\s*(?:lambda target:\s*)?\w+\.(make_\w+)', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("TILELANG_CACHE_DIR", "/work/tilelang_cache"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    root = Path(a.cache)
    assert root.is_dir(), f"no cache at {root}"
    per: dict[str, list[dict]] = defaultdict(list)
    files = 0
    for p in root.rglob("device_kernel.cu"):
        try:
            src = p.read_text()
        except OSError:
            continue
        files += 1
        row = _classify(src)
        if row:
            per[row.pop("name")].append(row)
    sites, reg = _call_sites(), _registry_names()

    print(f"# {files} cached device_kernel.cu, {len(per)} distinct kernels, cache {root}")
    print(f"# {'kernel':<26}{'wgmma':>6}{'mma.sync':>9}{'no MMA':>8}{'consumer':>18}  call sites")
    rows = []
    for name in sorted(per, key=lambda n: (-sum(r["cls"] == "mma.sync" for r in per[n]), n)):
        cells = per[name]
        n_w = sum(r["cls"] == "wgmma" for r in cells)
        n_m = sum(r["cls"] == "mma.sync" for r in cells)
        cons = sorted({r["consumer"] for r in cells if r["consumer"]})
        # a kernel is reached through its registry key, which may differ from the compiled name
        keys = [k for k in sites if k == name or reg.get(k, "").startswith(f"make_{name}")]
        site = "; ".join(dict.fromkeys(s for k in keys for s in sites[k])) \
            or "(not launched from backend.py)"
        print(f"  {name:<26}{n_w:>6}{n_m:>9}{len(cells) - n_w - n_m:>8}{str(cons):>18}  {site}")
        rows.append({"kernel": name, "wgmma": n_w, "mma_sync": n_m,
                     "no_mma": len(cells) - n_w - n_m, "consumer_widths": cons,
                     "call_sites": site})

    # only a kernel that ISSUES mma.sync is a candidate: a 64-wide consumer in a kernel with no
    # matrix instruction at all (rope, silu_mul, write_tokens) has nothing to widen.
    narrow = [r for r in rows if r["mma_sync"] and 64 in r["consumer_widths"]]
    print(f"\n# {len(narrow)} kernels emit mma.sync from a 64-wide consumer, which cannot issue "
          f"wgmma. Sorted by how many such cells:")
    for r in sorted(narrow, key=lambda r: -r["mma_sync"]):
        print(f"#   {r['kernel']}: {r['mma_sync']} mma.sync / {r['wgmma']} wgmma / "
              f"{r['no_mma']} no-MMA -- {r['call_sites']}")
    print("# A row here is a candidate, not a finding. A kernel with cells on BOTH sides is "
          "already reaching wgmma at some shapes, so the mma.sync cells may be a tile choice "
          "rather than a thread count; only a measurement per kernel decides (fp4_bwd 1.933x).")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
