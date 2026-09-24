#!/usr/bin/env python3
"""Parse nsys --stats cuda_gpu_kern_sum into kernel-class shares.

nsys 2022.4 on the V100 cannot export traces, so the capture ran with
`nsys profile --stats=true` and its stdout (this file's input) already carries
the table. Fold every kernel row into:

  gemm_fp4   *linear*/*gemm*/*gemv*/*mma*  (the NVFP4 linears; bf16 gemm rows,
             if any appear, are listed in the raw dump — names do not carry a
             dtype tag, so the fold is by name, checked against the dump)
  attn       *paged_attention*, *softmax*
  gdn        *gdn*
  norm_rope  *rmsnorm*, *rope*, *silu*
  other      anything else (memcpy/memset/casts/sampler/...)

Output: per-class count, total ms, share, plus the top 40 raw rows so the
name->class fold is auditable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

CLASSES = ("gemm_fp4", "attn", "gdn", "norm_rope", "other")


def classify(name: str) -> str:
    n = name.lower()
    if "gdn" in n:
        return "gdn"
    if "paged_attention" in n or "softmax" in n:
        return "attn"
    if any(k in n for k in ("linear", "gemm", "gemv", "mma")):
        return "gemm_fp4"
    if any(k in n for k in ("rmsnorm", "rope", "silu")):
        return "norm_rope"
    return "other"


def parse(text: str) -> list[tuple[str, int, int]]:
    """Return (name, total_ns, instances) rows from cuda_gpu_kern_sum."""
    rows = []
    sec = text.split("CUDA GPU Kernel Summary", 1)
    if len(sec) < 2:
        sec = text.split("cuda_gpu_kern_sum", 1)
    if len(sec) < 2:
        raise SystemExit("cuda_gpu_kern_sum section not found in nsys output")
    body = sec[1].split("\n\n", 1)[0]
    for ln in body.splitlines():
        # Numeric-prefixed rows: time%, total ns, instances, avg, med, ... name
        m = re.match(r"\s*[\d.]+\s+([\d,]+)\s+(\d+)\s+[\d,]+\s+[\d,]+", ln)
        if not m:
            continue
        total_ns = int(m.group(1).replace(",", ""))
        inst = int(m.group(2))
        name = ln[m.end():].strip()
        if name:
            rows.append((name, total_ns, inst))
    return rows


def mem_op_ms(text: str) -> float:
    """Total H2D/D2H copy time from the memory-operations summary. Captured
    graph staging copies land here, not in the kernel table."""
    sec = text.split("CUDA Memory Operation Summary", 1)
    if len(sec) < 2:
        sec = text.split("cuda_gpu_mem_time_sum", 1)
    if len(sec) < 2:
        return 0.0
    total = 0
    body = sec[1].split("\n\n", 1)[0]
    for ln in body.splitlines():
        m = re.match(r"\s*[\d.]+\s+([\d,]+)\s+(\d+)", ln)
        if m:
            total += int(m.group(1).replace(",", ""))
    return total / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nsys_log")
    ap.add_argument("--roof-ms", type=float, default=0.0,
                    help="weight-bytes/bandwidth roofline, printed for reference")
    ap.add_argument("--ticks", type=int, default=0,
                    help="captured tick count: normalize totals to ms/tick")
    ap.add_argument("--min-per-tick", type=int, default=0,
                    help="rc14 if total kernel instances per tick is below this "
                         "(graph-trace positive control: node tracing must expand "
                         "a 64-layer replay to several hundred kernels per tick)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.nsys_log, errors="replace") as f:
        text = f.read()
    rows = parse(text)
    if not rows:
        print("no kernel rows parsed — check the table format", file=sys.stderr)
        return 14
    agg = {c: {"count": 0, "ms": 0.0} for c in CLASSES}
    raw = []
    for name, ns, inst in rows:
        c = classify(name)
        agg[c]["count"] += inst
        agg[c]["ms"] += ns / 1e6
        raw.append({"name": name, "class": c, "instances": inst, "ms": round(ns / 1e6, 4)})
    total_ms = sum(a["ms"] for a in agg.values())
    mem_ms = mem_op_ms(text)
    for c in CLASSES:
        agg[c]["ms"] = round(agg[c]["ms"], 3)
        agg[c]["share"] = round(agg[c]["ms"] / total_ms, 4)
    raw.sort(key=lambda r: -r["ms"])
    total_instances = sum(a["count"] for a in agg.values())
    per_tick_n = total_instances / args.ticks if args.ticks else 0
    result = {"total_kernel_ms": round(total_ms, 3),
              "memcpy_ms": round(mem_ms, 3),
              "ticks": args.ticks,
              "kernel_instances": total_instances,
              "instances_per_tick": round(per_tick_n, 1),
              "roofline_ms": args.roof_ms, "classes": agg, "top40": raw[:40]}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    per = f" over {args.ticks} ticks = {total_ms / args.ticks:.2f} ms/tick" if args.ticks else ""
    print(f"total kernel time {total_ms:.2f} ms{per}"
          + (f" (roofline {args.roof_ms:.2f} ms)" if args.roof_ms else "")
          + (f"; memcpy {mem_ms:.2f} ms" if mem_ms else "")
          + f"; {total_instances} kernel instances"
          + (f" = {per_tick_n:.0f}/tick" if args.ticks else ""))
    for c in CLASSES:
        a = agg[c]
        tail = f" ({a['ms'] / args.ticks:.2f}/tick)" if args.ticks else ""
        print(f"{c:>10}: {a['ms']:>9.2f} ms {a['share']*100:>6.2f}%  x{a['count']}{tail}")
    if args.min_per_tick and per_tick_n < args.min_per_tick:
        print(f"FATAL only {per_tick_n:.0f} kernels/tick < {args.min_per_tick}: "
              "cuda graph node tracing not active (kernels collapsed into graph entry)",
              file=sys.stderr)
        return 14
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
