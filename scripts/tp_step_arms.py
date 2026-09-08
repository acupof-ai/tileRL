"""One 27B GRPO training step in two arms — tp=1 on one card, tp=2 under torchrun — so the
collectives' share of the step is readable, both measured in one session on one tree.

Four `prof_backward_ops.py` runs, because two numbers are wanted per arm and no single run
gives both: instrumented for the per-op table, then `--no-instrument` for the honest wall
clock (instrument() syncs before and after every handler, which removes the overlap the
shipped path gets, so a timed collective row is an upper bound).

Two things the per-op table cannot see, reported beside it rather than papered over:

* the optimizer's TP reduction (`train.py:174` sets `optimizer.tp_reduce = backend.all_reduce`)
  runs outside `tape.backward` and `backward_secs` subtracts it (`train.py:320`), so it is in
  no table row. `opt_s` below recovers it as `train_secs - backward_secs`.
* `coll%` is an upper bound, for the sync reason above. The bare arm's step is the wall clock.

    TILERL_TARGET=cpu python3 scripts/tp_step_arms.py --dry-run   # no GPU, tiny model
    python3 scripts/tp_step_arms.py --print-pod                   # the two-card command
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROF = ROOT / "scripts" / "prof_backward_ops.py"


def tree_id() -> str:
    """A hash of the code the arms execute, not HEAD: this is a SHARED checkout, so a peer's
    commit moves HEAD mid-run without changing what ran, and the profiler runs uncommitted."""
    h = hashlib.sha256()
    src = [PROF, *sorted((ROOT / "src" / "tilerl").rglob("*.py")),
           *sorted((ROOT / "packages/tilerl-kernels/src/tilerl_kernels").rglob("*.py"))]
    for p in src:
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def head_sha() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"


def prof_flags(a, tp: int) -> list[str]:
    """The `grpo-gsm8k-27b` settings, identical in both arms — only --tp may differ."""
    return [str(PROF), "--model", a.model, "--tp", str(tp), "--gen", str(a.gen),
            "--group", str(a.group), "--micro", "1", "--rank", "16",
            "--prompt-tokens", str(a.prompt_tokens), "--blocks", str(a.blocks),
            "--steps", str(a.steps)]


def claim_card(pid: int) -> None:
    """Re-claim the card for THIS arm's pid, the rule pod_run.sh states for a multi-arm wrapper.

    Four arms means three windows where the wrapper's claim reads stale and the card reads
    orphan, and an orphan card is what gets a container restarted under someone else's run.
    """
    if "BASH_FUNC_pod_run_claim%%" not in os.environ:
        return
    subprocess.run(["bash", "-c", f"pod_run_claim {pid}"], check=False)


def run_arm(a, name: str, tp: int, ranks: int, instrument: bool, outdir: Path) -> dict:
    out = outdir / f"{name}.json"
    for stale in outdir.glob(f"{name}.json*"):
        stale.unlink()
    flags = prof_flags(a, tp) + ([] if instrument else ["--no-instrument"])
    if ranks == 1:
        cmd = [sys.executable, "-u", *flags, "--out", str(out)]
    else:
        # --out is one path, so each rank suffixes its own or the ranks race on one file
        inner = shlex.join([sys.executable, "-u", *flags, "--out"])
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={ranks}",
               f"--master_port={a.master_port}", "--no-python", "bash", "-c",
               f"exec {inner} {shlex.quote(str(out))}.rank$RANK"]
    print(f"\n### {name}: {shlex.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT)
    claim_card(proc.pid)
    rc = proc.wait(timeout=a.timeout)
    paths = sorted(outdir.glob(f"{name}.json*"))
    if rc or len(paths) != ranks:
        raise SystemExit(f"{name}: exited {rc} with {len(paths)} of {ranks} result files")
    last = [json.loads(p.read_text())["rows"][-1] for p in paths]
    coll = [r for r in json.loads(paths[0].read_text()).get("table", [])
            if r["kind"] == "collective"]
    return {
        "name": name, "tp": tp, "ranks": ranks, "instrumented": instrument, "tree": tree_id(),
        # max, not rank 0: the step's wall clock is the slowest rank's
        "train_secs": max(r["train_secs"] for r in last),
        "backward_secs": max(r["backward_secs"] for r in last),
        "coll_secs": sum(r["secs"] for r in coll),
        "coll_calls": sum(r["calls"] for r in coll),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU tiny model end to end: the whole flow with no GPU")
    ap.add_argument("--print-pod", action="store_true", help="print the pod commands and exit")
    # No --ranks: world == tp by construction. prof_backward_ops.py takes only --tp
    # (`:490`) and threads it into _build_model, so it has no dp axis at all -- a world
    # wider than tp would profile a (dp, tp) mesh nothing under this script can build.
    ap.add_argument("--tp", type=int, default=2, help="the arm's shard width")
    ap.add_argument("--model", default="")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2, help="step 0 pays the JIT; the last is warm")
    ap.add_argument("--master-port", type=int, default=29571)
    ap.add_argument("--out-dir", default="/tmp/tp_step_arms")
    ap.add_argument("--timeout", type=int, default=7200, help="seconds per arm")
    a = ap.parse_args()
    tiny = a.dry_run
    a.model = a.model or ("tiny" if tiny else "qwen38-27b")
    a.gen = a.gen or (8 if tiny else 256)
    a.group = a.group or (2 if tiny else 8)
    a.prompt_tokens = a.prompt_tokens or (8 if tiny else 256)
    a.blocks = a.blocks or (64 if tiny else 4096)

    arm = f"tp{a.tp}x{a.tp}"
    if a.print_pod:
        driver = shlex.join(["python3", "-u", "scripts/tp_step_arms.py",
                             "--out-dir", "/work/tpstep", "--tp", str(a.tp)])
        print("# both arms in ONE pod job, so they share a session and a tree:")
        # 0,6 are tileRL's cards; 1-5 and 7 are aupai's and are never taken
        print(f"scripts/pod_run.sh tpstep 0,6 -- {driver}")
        print("# TILERL_TARGET=cuda, PYTHONPATH and the weights path come from pod_run.sh")
        print("tn exec 'tail -f /work/tpstep.log'   # the four arms stream here")
        return 0

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs: dict[tuple[str, bool], dict] = {}
    for name, tp, ranks in (("control", 1, 1), (arm, a.tp, a.tp)):
        for instr in (True, False):
            runs[(name, instr)] = run_arm(
                a, f"{name}-{'instr' if instr else 'bare'}", tp, ranks, instr, outdir)

    print(f"\n# {a.model} group={a.group} gen={a.gen} micro=1 rank=16 steps={a.steps}, "
          f"seed and data identical in both arms")
    print(f"# tree {tree_id()} (a hash of src + kernels + the profiler), HEAD {head_sha()}")
    print(f"# {'arm':<10} {'ranks':>5} {'tp':>3} {'step_s':>9} {'bwd_s':>9} {'opt_s':>9} "
          f"{'i-step_s':>9} {'i-coll_s':>9} {'coll':>6} {'coll%':>7}")
    for name in ("control", arm):
        bare, instr = runs[(name, False)], runs[(name, True)]
        opt = bare["train_secs"] - bare["backward_secs"]
        pct = instr["coll_secs"] / instr["train_secs"] * 100 if instr["train_secs"] else 0.0
        print(f"  {name:<10} {bare['ranks']:5d} {bare['tp']:3d} {bare['train_secs']:9.3f} "
              f"{bare['backward_secs']:9.3f} {opt:9.3f} {instr['train_secs']:9.3f} "
              f"{instr['coll_secs']:9.3f} {instr['coll_calls']:6d} {pct:6.2f}%")
    ctl, tpa = runs[("control", False)], runs[(arm, False)]
    if ctl["train_secs"]:
        print(f"# {arm}/control step {tpa['train_secs'] / ctl['train_secs']:.2f}x, "
              f"backward {tpa['backward_secs'] / max(ctl['backward_secs'], 1e-9):.2f}x")
    print("# opt_s is DERIVED as train_secs - backward_secs: rl_step subtracts optimizer_secs "
          "from backward_secs (train.py:320) and prof_backward_ops emits neither the key nor a "
          "table row for it, so TP's optimizer all_reduce (train.py:174) is ONLY here. On CUDA "
          "it also carries the profiler's trailing torch.cuda.synchronize, so it is an upper "
          "bound on the optimizer alone.")
    print("# i-coll_s and coll% are UPPER BOUNDS from the instrumented arm: instrument() syncs "
          "before and after each handler and removes the overlap the shipped path gets. step_s "
          "and bwd_s are the bare arm's, which is the honest wall clock.")
    bad = []
    trees = {r["tree"] for r in runs.values()}
    if len(trees) > 1:
        bad.append(f"the arms ran on DIFFERENT code {sorted(trees)} -- a peer edited this "
                   f"shared checkout mid-run and the arms are not comparable")
    if not runs[(arm, True)]["coll_calls"]:
        bad.append(f"the {arm} arm timed ZERO collective calls: --tp did not take effect and "
                   f"each rank profiled an unsharded step. The comparison is vacuous.")
    if bad:
        for line in bad:
            print(f"\nREFUSED: {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
