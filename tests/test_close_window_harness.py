"""Hermetic gates for scripts/run_close_window_v100.sh.

The window harness boots a 27B serve, so no test here runs an arm. What can be
checked without a card is the part that silently wastes a window when it is
wrong:

1. every arm's env delta names variables the source actually reads (a typo'd
   flag boots a serve that ignores it and reports the baseline twice);
2. the instrumentation preset is what the 2026-09-20 window needed, LIVENESS
   included — a 60 s liveness probe sends real chats into the measured window;
3. the arm list is stable and the unmerged #746 arm is refused, not run as a
   silent no-op;
4. arguments are parsed before any arm work, and bad input exits 2.

These run on macOS too: the script is orchestration over `env`/`curl` and does
no locking of its own.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

SRC = pathlib.Path(__file__).parent.parent / "scripts" / "run_close_window_v100.sh"
REPO = SRC.parent.parent


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SRC), *args], capture_output=True, text=True, timeout=60)


def _arm_envs() -> dict[str, str]:
    r = _run("--arms-env")
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.splitlines():
        if "|" in line:
            name, env = line.split("|", 1)
            out[name] = env.strip()
    return out


def test_every_arm_env_var_is_read_by_the_source():
    """The failure this prevents is silent: an unrecognized env var makes the arm
    identical to the one before it, so a window reports two baselines and a
    clean-looking delta of zero."""
    text = "\n".join(p.read_text() for p in
                     list((REPO / "src").rglob("*.py")) + list((REPO / "scripts").rglob("*.py")))
    seen = set()
    for env in _arm_envs().values():
        for tok in env.split():
            if "=" in tok:
                seen.add(tok.split("=", 1)[0])
    seen.update(t.split("=", 1)[0] for t in _run("--instrument-env").stdout.split() if "=" in t)
    assert seen, "no env vars extracted; the introspection surface changed"
    missing = sorted(v for v in seen if v not in text)
    assert not missing, f"harness sets env vars nothing reads: {missing}"


def test_the_instrumentation_preset_disables_liveness_polling():
    """LIVENESS_POLL_S is the documented 2026-09-19 mistake: the supervisor's
    liveness probe sends a real chat every 60 s and lands it in the decode
    window under measurement."""
    env = _run("--instrument-env").stdout.split()
    assert "LIVENESS_POLL_S=999999" in env, env
    assert "TILERL_STEP_TIMING=1" in env, env
    assert "TILERL_STEP_TIMING_SLOW_MS=0" in env, env


def test_arm_list_is_stable_and_the_unmerged_arm_is_refused():
    r = _run("--list")
    names = r.stdout.split()
    assert names == ["baseline", "batch", "bg1", "bg2", "bg3", "locksplit"], names
    # #746 is not merged: the arm must refuse rather than boot a serve that
    # ignores an unknown flag and report a no-op as a result.
    assert _arm_envs()["locksplit"] == "PENDING_746"
    src = SRC.read_text()
    assert "PENDING_746" in src and "SKIPPED" in src


def test_bg_arms_differ_only_in_the_depth_knob():
    """bg1/bg2/bg3 are the depth sweep. If two of them emit the same env the
    sweep measures one configuration three times."""
    arms = _arm_envs()
    assert arms["bg1"] != arms["bg2"] != arms["bg3"]
    for a in ("bg1", "bg2", "bg3"):
        assert "TILERL_CLOSE_BG_PUBLISH=1" in arms[a], (a, arms[a])
    # bg2 leaves the depth unset on purpose: build.py derives it from the shape.
    assert "TILERL_CLOSE_BG_DEPTH" not in arms["bg2"], arms["bg2"]


def test_baseline_arm_sets_no_experimental_flag():
    assert _arm_envs()["baseline"] == ""


def test_recovery_paths_exist_and_the_spill_delete_is_guarded():
    src = SRC.read_text()
    assert "--restore-only" in src
    # The spill is what a re-run needs and deleting it is irreversible, so the
    # cleanup must refuse while a serve is up and ask before each unlink.
    assert re.search(r"clean_spill\(\)", src)
    assert "REFUSING: a serve is still running" in src
    assert "rm -f" in src and "read -r ans" in src


def test_bad_argument_exits_2_before_any_arm_work():
    r = _run("--not-a-flag")
    assert r.returncode == 2, r.returncode
    assert "unknown arg" in r.stderr


def test_health_gate_checks_model_and_blocks_not_just_reachability():
    """/health answering 200 is not the same as the right server: a leftover
    tiny-model process or a different pool size answers too."""
    src = SRC.read_text()
    for needle in ("model=", "blocks_total", "EXPECT_BLOCKS", "EXPECT_MODEL"):
        assert needle in src, needle


def test_harness_reruns_the_log_through_the_standard_steady_filter():
    """The probe's own p50 is the wide set (dec>0). Without the re-filter, a
    headroom arm's number and a sweep arm's number look comparable and are not:
    the harness must produce the standard-set figure per arm."""
    src = SRC.read_text()
    assert "steady_filter.py" in src, "harness does not run the standard filter"
    assert "steady.json" in src
    # The filter's own contract, pinned where it is defined.
    sf = (REPO / "scripts" / "steady_filter.py").read_text()
    assert "path != graph" in sf and "TAIL_MS = 300" in sf


def test_steady_filter_splits_the_tail_by_threshold_not_quantile():
    """Executes the shipped self-check, then re-runs it against a synthetic log
    where a quantile cut provably fails: few ticks, one enormous. The assertion
    is that the enormous tick lands in tail_n -- a 0.95 quantile over 5 ticks
    would put it in the median."""
    import json
    import tempfile

    sf_script = REPO / "scripts" / "steady_filter.py"
    r = subprocess.run(["python3", str(sf_script), "--self-check"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-400:]

    log = "\n".join([
        "[step-timing] tick 1 total=29ms dec=0 pre=0 model=0ms path=graph sparse=0",
        "[step-timing] tick 2 total=176ms dec=1 pre=0 model=160ms sample=3ms path=eager sparse=1",
        "[step-timing] tick 3 total=181ms dec=1 pre=0 model=162ms sample=3ms path=eager sparse=1",
        "[step-timing] tick 4 total=179ms dec=1 pre=0 model=161ms sample=3ms path=eager sparse=1",
        "[step-timing] tick 5 total=5000ms dec=1 pre=0 model=4000ms sample=3ms path=eager sparse=1",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(log)
        path = fh.name
    r = subprocess.run(["python3", str(sf_script), "--log", path],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-400:]
    rec = json.loads(r.stdout)
    assert rec["steady_n"] == 3, rec
    assert rec["tail_n"] == 1 and rec["tail_max_ms"] == 5000, rec
    assert rec["steady_p50_ms"] in (179, 181), rec
    assert rec["excluded_path_graph_n"] == 1, rec
