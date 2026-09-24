#!/usr/bin/env python3
"""PROBE-ONLY #805: v2 lag-1 refresh CPU installation-correctness gate.

Drives scripts/probe_serve_sm70_w2048.build_smoke_engine (the tiny CPU model)
under TILERL_SPARSE_V2 = inline | async, each in its own subprocess, and proves
two things before any device window:

  1. INSTALLATION: inline and async lag-1 produce the SAME 160-token sequence
     (both drive the shared sf.select_refresh; async additionally runs the
     selection on post-replay staging snapshots, which is the only intended
     difference). A mismatch -> rc1.
  2. NEGATIVE CONTROL: with exactly ONE committed override phys page changed to
     a wrong block (monkeypatched from this gate, src untouched), the carry
     replay MUST attend different KV and diverge from the clean async run. If
     it still matches, the override is inert (the merge does not reach the
     attention table) -> rc1.

CPU-only, no card. Usage: uv run python -u scripts/probe_v2_install_gate.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_NEW = 160
OUTDIR = "/tmp/v2_install_gate"

WORKER = f'''
import sys, json, os
os.environ["TILERL_TARGET"] = "cpu"
sys.path.insert(0, {HERE!r})
from probe_serve_sm70_w2048 import build_smoke_engine, _sampling
mode, kind, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["TILERL_SPARSE_V2"] = mode
if kind == "corrupt":
    from tilerl.sparse_engine import SparseForward
    _arm = SparseForward.arm_override
    def _corrupt(self, per_group, use_rows):
        _arm(self, per_group, use_rows)
        t = self._ov_phys_t[0]
        if int(self._ov_nsel_t[0][0]) >= 1:
            t[0, 0] = 140  # one wrong committed phys page for row0/group0
    SparseForward.arm_override = _corrupt
e, _be, _cfg = build_smoke_engine("graph_w2048")
rid = e.submit([7 + (i % 300) for i in range(400)], _sampling({N_NEW}))
lag = e._sparse._lag()
for _ in range(200000):
    e.step()
    if not any(r.req_id == rid for r in e._running):
        break
out = e.poll().get(rid, [])
json.dump({{"output": out, "carry": lag.carry_cycles, "fb": lag.fallback_cycles}},
          open(outpath, "w"))
print(mode, kind, len(out), lag.carry_cycles, lag.fallback_cycles)
e.shutdown()
'''


def _first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def run_worker(mode, kind):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{mode}_{kind}.json")
    if os.path.exists(out):
        os.remove(out)
    with open(os.path.join(OUTDIR, f"{mode}_{kind}.log"), "w") as lf:
        rc = subprocess.run([sys.executable, "-u", "-c", WORKER, mode, kind, out],
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.exists(out):
        print(f"{mode}/{kind}: worker rc={rc}, no json -> rc14", file=sys.stderr)
        return 14, None
    return 0, json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rcs = {}
    inline = async_ = corrupt = None
    for mode, kind, slot in (("inline", "clean", "inline"),
                             ("async", "clean", "async"),
                             ("async", "corrupt", "corrupt")):
        rc, rep = run_worker(mode, kind)
        rcs[f"{mode}:{kind}"] = rc
        if rc:
            print(json.dumps(rcs, indent=2))
            return rc
        if slot == "inline":
            inline = rep
        elif slot == "async":
            async_ = rep
        else:
            corrupt = rep

    problems = []
    # Every carry must have been served by a prepared selection (no eager
    # fallback masquerading as a pass), and both modes carry equally.
    if inline["carry"] == 0 or inline["fb"] != 0 or async_["fb"] != 0:
        problems.append(
            f"carry/fb inline={inline['carry']}/{inline['fb']} "
            f"async={async_['carry']}/{async_['fb']} (need carry>0, fb=0)")
    fd_same = _first_div(inline["output"], async_["output"])
    if fd_same is not None:
        problems.append(f"inline vs async first divergence at {fd_same} (must be identical)")
    fd_corrupt = _first_div(async_["output"], corrupt["output"])
    if fd_corrupt is None:
        problems.append("one wrong committed phys page did NOT change output "
                        "(override is inert / merge does not reach attention)")
    verdict = {
        "inline_vs_async_first_div": fd_same,
        "corrupt_one_phys_page_first_div": fd_corrupt,
        "inline_carry": inline["carry"], "async_carry": async_["carry"],
        "fallback_cycles": [inline["fb"], async_["fb"], corrupt["fb"]],
        "n_tokens": len(inline["output"]),
    }
    with open(os.path.join(OUTDIR, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    if problems:
        for p in problems:
            print("RED: " + p, file=sys.stderr)
        print(json.dumps(verdict, indent=2))
        return 1
    print("v2 install gate GREEN")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
