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
import time

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
    # the boot self-certification line names the default arm, window included
    assert "arm: depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8 w=0" in out
    # Never the V100 hard-coded venv, and never launched through `uv run` (the
    # cu130 torch it resolves cannot load on the pod's 12.9 driver).
    assert "venv70" not in out
    assert argv[0] != "uv" and not argv[0].endswith("/uv"), f"launched through uv: {argv[0]}"


def test_the_arm_name_leads_the_banner_and_is_omitted_when_unset():
    # An inspector reads which arm a boot is from the boot line, so the name must
    # be IN it -- and first, since a truncated line loses its tail, not its head.
    rc, out, err = _dry_run({"SERVE_ARM_NAME": "A4"})
    assert rc == 0, err
    assert "arm: arm=A4 depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8 w=0" in out
    # Unset and empty are both "no name": appending a bare `arm=` would read as a
    # field with a missing value, and would also change the banner every existing
    # consumer of this line already parses.
    default = "arm: depth=1 sparse_k=128 decode_graph=on ctx=131072 slots=8 w=0"
    assert default in _dry_run({})[1]
    assert default in _dry_run({"SERVE_ARM_NAME": ""})[1]
    # The name is a log label, never an argv token: it must not reach the serve.
    assert "A4" not in _dry_run({"SERVE_ARM_NAME": "A4"})[1].split()


def test_decode_graph_off_arm_passes_the_explicit_force_off_flag():
    rc, out, err = _dry_run({"SERVE_DECODE_GRAPH": "0", "SERVE_DEPTH": "3"})
    assert rc == 0, err
    argv = out.split()
    # Off MUST be the explicit const=False flag, not an omission: CLI default None
    # AUTO-enables capture on sm90, so a missing flag would silently stay graph-on.
    assert "--no-decode-graph" in argv
    assert "--decode-graph" not in argv
    assert "arm: depth=3 sparse_k=128 decode_graph=off ctx=131072 slots=8 w=0" in out


def test_the_draft_window_reaches_argv_only_when_nonzero_and_self_reports():
    # A6's arm: the launched window must be in the resolved argv, or the arm is
    # silently the full-prefix one -- and the descriptor must name it, because
    # A6's evidence is that banner rather than an environ read.
    rc, out, err = _dry_run({"SERVE_DRAFT_WINDOW": "2048"})
    assert rc == 0, err
    argv = out.split()
    i = argv.index("--draft-attn-window-tokens")
    assert argv[i + 1] == "2048", argv
    assert "w=2048" in out
    # 0 IS the CLI default (full prefix), so the default arm adds no flag: an
    # explicit `--draft-attn-window-tokens 0` would be one more token to read
    # past in a log holding every arm.
    assert "--draft-attn-window-tokens" not in _dry_run({})[1].split()


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


def test_dry_run_names_the_per_arm_trace_file():
    rc, out, err = _dry_run({"SERVE_TRACE": "/work/cold_trace_a4.txt"})
    assert rc == 0, err
    assert "trace=/work/cold_trace_a4.txt" in out


TRACE_SRC = pathlib.Path(__file__).parent.parent / "scripts" / "serve_cold_trace.sh"


def test_cold_trace_sampler_dies_with_its_serve_pid(tmp_path):
    # The sampler's only termination condition is the tracked serve pid: a
    # standalone loop with a fixed iteration count died silent mid-matrix and
    # kept writing across arms. Bound to a live pid, it must exit when that pid
    # is gone (the curl against an absent /health adds nothing but must not end
    # the loop while the serve still lives).
    serve = subprocess.Popen(["sleep", "30"])
    try:
        samp = subprocess.Popen(
            ["bash", str(TRACE_SRC), str(tmp_path / "trace.txt"), str(serve.pid), "1"])
        time.sleep(2)
        assert serve.poll() is None and samp.poll() is None, "sampler died with serve alive"
        serve.terminate()
        serve.wait()
        for _ in range(30):
            if samp.poll() is not None:
                break
            time.sleep(0.5)
        assert samp.poll() is not None, "sampler outlived the serve it was bound to"
    finally:
        serve.kill()
        samp.kill()


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


def _guard_env(env_extra: dict[str, str]) -> str:
    """Run the real launcher with an interpreter that dumps its environment, and
    return that dump -- the environment serve_liveness.py would read.

    The launcher exports LIVENESS_POLL_S before spawning the serve, and the guard is
    a sibling under the same shell, so what the child receives is what the guard
    receives. `LIVENESS_POLL_S` is dropped from the inherited environment first: the
    "unset" case must be unset whatever this machine happens to carry."""
    d, env = _fuse_sandbox(env_extra)
    env.pop("LIVENESS_POLL_S", None)
    env.update(env_extra)
    stub = d / "dumppy"
    stub.write_text('#!/bin/bash\nexport > "$SERVE_LOG.childenv"\nexit 7\n')
    stub.chmod(0o755)
    env["SERVE_PYTHON"] = str(stub)
    subprocess.run(["bash", str(SRC)], capture_output=True, text=True, timeout=120, env=env)
    return (d / "serve.log.childenv").read_text()


def test_the_guard_poll_period_is_passed_through_and_defaults_to_60():
    """LIVENESS_POLL_S must reach the guard, and unset must stay 60 -- the shipped
    poll period. At slots=8 the guard injects a real 4-token chat every poll, which
    is what poisoned a zero-traffic baseline, so the override has to survive the
    launcher rather than being clobbered by it."""
    # Unset: the guard's own default (serve_liveness.py) is 60, and the launcher
    # must not turn that into anything else.
    env_txt = _guard_env({})
    assert 'LIVENESS_POLL_S="60"' in env_txt, [ln for ln in env_txt.splitlines()
                                               if "LIVENESS" in ln]
    # A caller's value wins -- this is the zero-traffic-baseline override.
    env_txt = _guard_env({"LIVENESS_POLL_S": "999999"})
    assert 'LIVENESS_POLL_S="999999"' in env_txt, [ln for ln in env_txt.splitlines()
                                                   if "LIVENESS" in ln]
    # An EMPTY value is the one case the export changes: float('') would raise in
    # the guard, so it falls back to 60 instead of crashing the liveness loop.
    env_txt = _guard_env({"LIVENESS_POLL_S": ""})
    assert 'LIVENESS_POLL_S="60"' in env_txt, [ln for ln in env_txt.splitlines()
                                               if "LIVENESS" in ln]


def test_the_header_names_the_argv_prefix_route_not_a_caller_export():
    """pod_run bakes the CMD into a runner executed inside the container and does not
    forward the caller's environment, so `export LIVENESS_POLL_S=... ` on the laptop
    never arrives. The header has to say so or the next person loses the same window."""
    text = SRC.read_text()
    assert "LIVENESS_POLL_S (60)" in text, "the header does not document the knob"
    assert "LIVENESS_POLL_S=999999 bash scripts/serve_h20.sh" in text, text[-600:]
    assert "argv PREFIX" in text and "does not" in text
