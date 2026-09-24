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
from types import SimpleNamespace as _S

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
_promo0 = int(e._sparse.ctx.kv.cold.promotions)

if kind in ("b2guard", "b2noguard"):
    # 94(a) negative control: two concurrent prompts drive B=2 graph verify
    # ticks (the CUDA illegal-access shape). With the guard on, commit must
    # never arm a B=2 carry; with enforce_b1 flipped off, a B=2 carry MUST arm
    # (proving the guard is the only thing blocking it).
    from tilerl.sparse_lag import LagController
    lag = e._sparse._lag()
    lag.enforce_b1 = (kind == "b2guard")
    _orig_commit = LagController.commit
    armed_rows = []
    def _wc(self, sf, srows):
        ok = _orig_commit(self, sf, srows)
        if ok:
            armed_rows.append(len(srows))
        return ok
    LagController.commit = _wc
    tick_bs = []
    _orig_rdg = e._sparse.run_decode_graph
    def _wrdg(reqs, chains=None):
        tick_bs.append((len(reqs),
                        max((len(c) for c in chains), default=1) if chains else 1))
        return _orig_rdg(reqs, chains)
    e._sparse.run_decode_graph = _wrdg
    rids = [e.submit([7 + (i % 300) for i in range(L)], _sampling({N_NEW}))
            for L in (400, 300)]
    done = {{}}
    for _ in range(400000):
        e.step()
        done.update(e.poll())  # a live row can briefly leave _running (waiting)
        if all(rid in done for rid in rids):
            break
    outs = [done.get(rid, []) for rid in rids]
    b2_verify_ticks = sum(1 for b, w in tick_bs if b == 2 and w > 1)
    json.dump({{"lengths": [len(o) for o in outs],
               "b2_verify_ticks": b2_verify_ticks,
               "armed_b2": armed_rows.count(2),
               "armed_b1": armed_rows.count(1),
               "b1_guard_fallbacks": lag.b1_guard_fallbacks,
               "carry": lag.carry_cycles, "fb": lag.fallback_cycles}},
              open(outpath, "w"))
    print(kind, [len(o) for o in outs], "b2vt", b2_verify_ticks,
          "armed", armed_rows, "guardfb", lag.b1_guard_fallbacks)
    e.shutdown()
    sys.exit(0)

if kind in ("tfclean", "tfbuggy"):
    # impl-39's graph-path defect: SparseCtx.verify is a bound method captured
    # at build; patching e._verify alone leaves graph verify ticks on the
    # original, so rejected draft slots stay accepted=True. Count PRODUCTION
    # rejects independently (output growth per verify tick), then compare with
    # the recorder's accepted=False marks. tfbuggy re-creates the defect by
    # restoring ctx.verify after install.
    from probe_teacher_force import TeacherForceRecorder
    prod = {{"verify_calls": 0, "reject_slots": 0}}
    _raw_verify = e._verify
    def _counting_verify(rows, chains, logits, hidden):
        before = [len(r.output) for r in rows]
        _raw_verify(rows, chains, logits, hidden)
        for i, r in enumerate(rows):
            if len(chains[i]) > 1:
                prod["verify_calls"] += 1
                # committed tokens this tick minus the mandatory bonus token =
                # number of DRAFT slots accepted; the rest were rejected.
                growth = len(r.output) - before[i]
                prod["reject_slots"] += (len(chains[i]) - 1) - (growth - 1)
    e._verify = _counting_verify
    object.__setattr__(e._sparse.ctx, "verify", _counting_verify)
    rec = TeacherForceRecorder(e, None).install()
    if kind == "tfbuggy":
        object.__setattr__(e._sparse.ctx, "verify", rec._orig_ctx_verify)
    patched_ctx = (e._sparse.ctx.verify is rec._patched_verify)
    rid = e.submit([7 + (i % 300) for i in range(400)], _sampling({N_NEW}))
    done = {{}}
    for _ in range(200000):
        e.step()
        done.update(e.poll())
        if rid in done:
            break
    out = done.get(rid, [])
    allrows = rec.rows.get(0, [])
    rej = [x for x in allrows if x.get("accepted") is False]
    acc = rec.accepted_positions(0)
    rec.uninstall()
    ctx_restored = (e._sparse.ctx.verify is _counting_verify)
    json.dump({{"n_output": len(out), "verify_calls": prod["verify_calls"],
               "prod_reject_slots": prod["reject_slots"],
               "recorder_reject_slots": len(rej),
               "accepted_rows": len(acc),
               "patched_ctx_at_run": patched_ctx,
               "ctx_restored_after_uninstall": ctx_restored}},
              open(outpath, "w"))
    print(kind, "vcalls", prod["verify_calls"], "prodrej",
          prod["reject_slots"], "recrej", len(rej), "acc", len(acc),
          "out", len(out), "patched_ctx", patched_ctx)
    e.shutdown()
    sys.exit(0)

rid = e.submit([7 + (i % 300) for i in range(400)], _sampling({N_NEW}))
lag = e._sparse._lag()
for _ in range(200000):
    e.step()
    if not any(r.req_id == rid for r in e._running):
        break
out = e.poll().get(rid, [])
json.dump({{"output": out, "carry": lag.carry_cycles, "fb": lag.fallback_cycles,
            "cold_promos": int(e._sparse.ctx.kv.cold.promotions) - _promo0}},
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
    with open(out) as jf:
        return 0, json.load(jf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="",
                    help="comma list of scenario groups to skip "
                         "(clean|b2|tf|shared); default runs all")
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)

    # Shared-prefix carry regression: a prefix-adopting follower's early pages
    # resolve through the shared tier, not the private cold tier; the lag job
    # must promote them and arm the carry. Independent script, real engine.
    if "shared" not in skip:
        gate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "probe_v2_shared_carry_gate.py")
        src = subprocess.run([sys.executable, "-u", gate],
                             capture_output=True, text=True)
        if src.returncode != 0:
            print(src.stdout[-1000:])
            print(src.stderr[-2000:], file=sys.stderr)
            print("RED: shared-prefix carry gate failed", file=sys.stderr)
            return 1
        print(src.stdout.strip().splitlines()[-1])
    rcs = {}
    inline = async_ = corrupt = None
    b2guard = b2noguard = tfclean = tfbuggy = None
    jobs = [(("inline", "clean", "inline"), "clean"),
            (("async", "clean", "async"), "clean"),
            (("async", "corrupt", "corrupt"), "clean"),
            (("async", "b2guard", "b2guard"), "b2"),
            (("async", "b2noguard", "b2noguard"), "b2"),
            (("off", "tfclean", "tfclean"), "tf"),
            (("off", "tfbuggy", "tfbuggy"), "tf")]
    for (mode, kind, slot), grp in jobs:
        if grp in skip:
            continue
        rc, rep = run_worker(mode, kind)
        rcs[f"{mode}:{kind}"] = rc
        if rc:
            print(json.dumps(rcs, indent=2))
            return rc
        if slot == "inline":
            inline = rep
        elif slot == "async":
            async_ = rep
        elif slot == "corrupt":
            corrupt = rep
        elif slot == "b2guard":
            b2guard = rep
        elif slot == "b2noguard":
            b2noguard = rep
        elif slot == "tfclean":
            tfclean = rep
        else:
            tfbuggy = rep

    problems = []
    fd_same = fd_corrupt = _coldval = None
    skips_normal = skips_carry = skips_cpu = buggy_skips_carry = None
    if "clean" not in skip:
        # Every carry must have been served by a prepared selection (no eager
        # fallback masquerading as a pass), and both modes carry equally.
        if inline["carry"] == 0 or inline["fb"] != 0 or async_["fb"] != 0:
            problems.append(
                f"carry/fb inline={inline['carry']}/{inline['fb']} "
                f"async={async_['carry']}/{async_['fb']} (need carry>0, fb=0)")
        # The async worker must ACTUALLY drive cold take/reserve/H2D; if every
        # pick were already resident the whole promote path would be untested.
        if async_.get("cold_promos", 0) <= 0:
            problems.append(
                f"async cold_promos={async_.get('cold_promos')} — the worker "
                f"never took/promoted a cold page, so the reserve/H2D path is "
                "untested")
        fd_same = _first_div(inline["output"], async_["output"])
        if fd_same is not None:
            problems.append(
                f"inline vs async first divergence at {fd_same} (must be identical)")
        fd_corrupt = _first_div(async_["output"], corrupt["output"])
        if fd_corrupt is None:
            problems.append("one wrong committed phys page did NOT change output "
                            "(override is inert / merge does not reach attention)")
        _coldval = async_.get("cold_promos")

    # CUDA-branch reconcile predicate. On CPU the device.type=="cuda" clause is
    # always False, which hid rev's dead refresh_tick flag. Force a CUDA-like
    # sf and assert: an ordinary captured tick skips reconcile, a carry does
    # not. Negative control: a predicate missing the refresh_tick exception
    # wrongly skips the carry — the exact bug — and this check catches it.
    if "clean" not in skip:
        from tilerl.sparse_runtime import _captured_skips_pin_reconcile as pred

        cuda_sf = _S(device_select=True, device=_S(type="cuda"))
        cuda_carry = _S(device_select=True, refresh_tick=True,
                        device=_S(type="cuda"))
        skips_normal = pred(cuda_sf)
        skips_carry = pred(cuda_carry)
        cpu_carry = _S(device_select=True, refresh_tick=True,
                       device=_S(type="cpu"))
        skips_cpu = pred(cpu_carry)
        # Negative control of THIS check: the buggy predicate (the pre-fix
        # code, which ignored refresh_tick) returns True for a carry, proving
        # the skips_carry assertion below would have fired on the real blocker.
        buggy_skips_carry = (getattr(cuda_carry, "device_select", False)
                             and cuda_carry.device.type == "cuda")
        if not buggy_skips_carry:
            problems.append("reconcile negative control broken: the pre-fix "
                            "predicate should skip the carry (so the gate can "
                            "distinguish it)")
        if not skips_normal:
            problems.append("reconcile predicate: ordinary captured CUDA tick "
                            "must skip pin reconcile")
        if skips_carry:
            problems.append("reconcile predicate: a v2 carry (refresh_tick) "
                            "must NOT skip pin reconcile — rev's dead-flag blocker")
        if skips_cpu:
            problems.append("reconcile predicate: CPU must never skip reconcile")

    # 94(a) B>1 guard. The scenario must actually contain B=2 W=2 graph verify
    # ticks or the guard is untested. With the guard, ZERO B=2 carries arm and
    # every B=2 prepare is counted. With enforce_b1 flipped off, at least one
    # B=2 carry MUST arm — proving the guard is the sole thing blocking it.
    if "b2" not in skip:
        if b2guard["b2_verify_ticks"] <= 0 or b2noguard["b2_verify_ticks"] <= 0:
            problems.append(
                f"B=2 negative control untested: b2_verify_ticks "
                f"guard={b2guard['b2_verify_ticks']} "
                f"noguard={b2noguard['b2_verify_ticks']} (need >0)")
        if b2guard["armed_b2"] != 0:
            problems.append(
                f"B>1 guard failed: {b2guard['armed_b2']} B=2 carries armed "
                "with enforce_b1 on")
        if b2guard["b1_guard_fallbacks"] <= 0:
            problems.append(
                f"B>1 guard never engaged: b1_guard_fallbacks="
                f"{b2guard['b1_guard_fallbacks']} despite "
                f"{b2guard['b2_verify_ticks']} B=2 verify ticks")
        if b2noguard["armed_b2"] <= 0:
            problems.append(
                "B>1 negative control broken: with enforce_b1 flipped off no "
                "B=2 carry armed — the gate would pass even if the guard were "
                "removed")
        if b2noguard["b1_guard_fallbacks"] != 0:
            problems.append(
                f"enforce_b1 off still counted guard refusals: "
                f"{b2noguard['b1_guard_fallbacks']}")

    # impl-39 graph-path verify binding. Production output growth gives an
    # independent count of rejected draft slots; the recorder must report the
    # same number. tfbuggy re-creates "ctx.verify left unpatched" and MUST show
    # the recorder undercounting (or the negative control is broken).
    if "tf" not in skip:
        if tfclean["verify_calls"] <= 0 or tfclean["prod_reject_slots"] <= 0:
            problems.append(
                f"teacher-force graph gate untested: verify_calls="
                f"{tfclean['verify_calls']} prod_reject_slots="
                f"{tfclean['prod_reject_slots']} (need a tick with a real reject)")
        if not tfclean["patched_ctx_at_run"]:
            problems.append("install() did not bind ctx.verify to the patched "
                            "verify (graph ticks bypass the recorder)")
        if tfclean["recorder_reject_slots"] != tfclean["prod_reject_slots"]:
            problems.append(
                f"recorder rejected slots {tfclean['recorder_reject_slots']} "
                f"!= production rejects {tfclean['prod_reject_slots']} on the "
                "graph verify path (accepted_positions still polluted)")
        # Each committed output position has exactly one accepted row; rejected
        # draft slots add extra rows. accepted == n_output is the invariant.
        if tfclean["accepted_rows"] != tfclean["n_output"]:
            problems.append(
                f"accepted_positions {tfclean['accepted_rows']} != outputs "
                f"{tfclean['n_output']} (each committed position must have "
                "exactly one accepted record)")
        if not tfclean["ctx_restored_after_uninstall"]:
            problems.append("uninstall() did not restore ctx.verify")
        if tfbuggy["patched_ctx_at_run"]:
            problems.append("tfbuggy setup failed: ctx.verify got patched")
        if tfbuggy["recorder_reject_slots"] >= tfbuggy["prod_reject_slots"] \
                and tfbuggy["prod_reject_slots"] > 0:
            problems.append(
                "teacher-force negative control broken: leaving ctx.verify "
                "unpatched did NOT undercount rejects (recorder "
                f"{tfbuggy['recorder_reject_slots']} >= prod "
                f"{tfbuggy['prod_reject_slots']})")

    verdict = {}
    if "clean" not in skip:
        verdict.update({
            "inline_vs_async_first_div": fd_same,
            "corrupt_one_phys_page_first_div": fd_corrupt,
            "inline_carry": inline["carry"], "async_carry": async_["carry"],
            "fallback_cycles": [inline["fb"], async_["fb"], corrupt["fb"]],
            "async_cold_promotions": _coldval,
            "reconcile_predicate": {
                "ordinary_cuda_captured_skips": bool(skips_normal),
                "v2_carry_skips": bool(skips_carry),
                "cpu_skips": bool(skips_cpu)},
            "n_tokens": len(inline["output"])})
    if "b2" not in skip:
        verdict["b1_guard"] = {
            "b2_verify_ticks": [b2guard["b2_verify_ticks"],
                                b2noguard["b2_verify_ticks"]],
            "armed_b2": [b2guard["armed_b2"], b2noguard["armed_b2"]],
            "guard_fallbacks": [b2guard["b1_guard_fallbacks"],
                                b2noguard["b1_guard_fallbacks"]]}
    if "tf" not in skip:
        verdict["teacher_force_graph_verify"] = {
            "clean": tfclean, "buggy_ctx_unpatched": tfbuggy}
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
