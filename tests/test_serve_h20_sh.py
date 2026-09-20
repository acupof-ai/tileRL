"""Hermetic gates for scripts/serve_h20.sh.

The H20 supervisor is run through pod_run.sh (which owns the card claim, the
cu129 interpreter on PATH and the pod-synced tree), so these gates never start a
server: `--dry-run` asserts the resolved serve argv, and the restart-fuse tests
drive the real script with a stub python that crashes immediately, so the
readiness guard exits on its first kill -0 and warmup/liveness never start.

Cross-platform on purpose: unlike serve_hybrid_v100.sh this script has no
flock(1) mutual exclusion (pod_run's one-live-name refusal is the guard), so the
gates run on the macOS CI row too.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile

SRC = pathlib.Path(__file__).parent.parent / "scripts" / "serve_h20.sh"


def _dry_run(env_extra: dict[str, str]) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["SERVE_PYTHON"] = shutil.which("python3") or "python3"
    env.update(env_extra)
    r = subprocess.run(["bash", str(SRC), "--dry-run"], capture_output=True,
                       text=True, timeout=30, env=env)
    return r.returncode, r.stdout, r.stderr


def test_dry_run_resolves_the_sparse_d1_decode_graph_argv():
    rc, out, err = _dry_run({})
    assert rc == 0, err
    argv = out.split()
    for flag in ("--sparse-k", "128", "--sparse-min-tokens", "8192",
                 "--depth", "1", "--decode-graph", "--cold-ssd-path"):
        assert flag in argv, f"missing {flag}: {argv}"
    # the boot self-certification line names the default arm
    assert "arm: depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8" in out
    # Never the V100 hard-coded venv, and never launched through `uv run` (the
    # cu130 torch it resolves cannot load on the pod's 12.9 driver).
    assert "venv70" not in out
    assert argv[0] != "uv" and not argv[0].endswith("/uv"), f"launched through uv: {argv[0]}"


def test_decode_graph_off_arm_passes_the_explicit_force_off_flag():
    rc, out, err = _dry_run({"SERVE_DECODE_GRAPH": "0", "SERVE_DEPTH": "3"})
    assert rc == 0, err
    argv = out.split()
    # Off MUST be the explicit const=False flag, not an omission: CLI default None
    # AUTO-enables capture on sm90, so a missing flag would silently stay graph-on.
    assert "--no-decode-graph" in argv
    assert "--decode-graph" not in argv
    assert "arm: depth=3 sparse_k=128 decode_graph=off ctx=131072 slots=8" in out


def test_dense_arm_keeps_decode_graph_off():
    rc, out, err = _dry_run({"SERVE_SPARSE_K": "0", "SERVE_DECODE_GRAPH": "0"})
    assert rc == 0, err
    argv = out.split()
    assert "--sparse-k" in argv and "0" in argv
    assert "--no-decode-graph" in argv and "--decode-graph" not in argv
    assert "decode_graph=off" in out and "sparse_k=0" in out


def test_an_empty_cold_path_drops_the_spill_tier_flags():
    rc, out, err = _dry_run({"SERVE_COLD_SSD": ""})
    assert rc == 0, err
    argv = out.split()
    assert "--cold-ssd-path" not in argv
    assert "--kv-cold-bytes" not in argv
    assert "cold_ssd=<disabled>" in out


def _fuse_sandbox(env_extra: dict[str, str]):
    d = pathlib.Path(tempfile.mkdtemp(prefix="serve_h20_fuse."))
    repo = d / "tree"
    (repo / "src").mkdir(parents=True)
    (repo / ".synced_commit").write_text("deadbeef\n")
    stub = d / "crashpy"
    stub.write_text("#!/bin/bash\nexit 7\n")
    stub.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "SERVE_REPO": str(repo),
        "SERVE_PYTHON": str(stub),
        "SERVE_LOG": str(d / "serve.log"),
        "SERVE_FUSE_STATE": str(d / ".fuse"),
        "SERVE_TMP": str(d / "tmp"),
        "MAX_RESTARTS": "10",
        "RESTART_FUSE_MAX": "2",
        "RESTART_FUSE_WINDOW_S": "600",
    })
    env.update(env_extra)
    return d, env


def test_a_crash_burst_trips_the_fuse_and_stays_down():
    d, env = _fuse_sandbox({})
    r = subprocess.run(["bash", str(SRC)], capture_output=True, text=True,
                       timeout=120, env=env)
    assert r.returncode == 2, r.stderr[:300]
    log = (d / "serve.log").read_text()
    assert "FUSE: 2 restarts within 600s" in log
    assert len((d / ".fuse").read_text().split()) == 2
    assert log.count("boot ") == 2


def test_crashes_outside_the_window_age_out_and_give_up_not_trip():
    d, env = _fuse_sandbox({"RESTART_FUSE_WINDOW_S": "1", "MAX_RESTARTS": "2"})
    r = subprocess.run(["bash", str(SRC)], capture_output=True, text=True,
                       timeout=120, env=env)
    assert r.returncode == 1, r.stderr[:300]
    log = (d / "serve.log").read_text()
    assert "FUSE:" not in log
    assert "gave up after 2 restarts" in log
