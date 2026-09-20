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
    # The draft READ window is instrumentation, not a treatment, and the probe
    # asserts it (--expect-window 2048). Without it the loader default is W=0 and
    # every arm exits rc13 before producing a number.
    assert "TILERL_DRAFT_ATTN_WINDOW_TOKENS=2048" in env, env


def test_arm_list_is_stable_and_the_unmerged_arm_is_refused():
    r = _run("--list")
    names = r.stdout.split()
    assert names == ["baseline", "batch", "bg1", "bg2", "bg3", "bgcap", "locksplit"], names
    # #746 is not merged: the arm must refuse rather than boot a serve that
    # ignores an unknown flag and report a no-op as a result.
    assert _arm_envs()["locksplit"] == "PENDING_746"
    src = SRC.read_text()
    assert "PENDING_746" in src and "SKIPPED" in src


def test_bgcap_is_the_only_arm_with_the_spill_cap():
    """The cap changes what the engine does, so it is an arm and not shared
    instrumentation: on every arm it would make them incomparable on the thing
    they are compared on. bg2 is its control (same bg config, no cap)."""
    arms = _arm_envs()
    assert "TILERL_COLD_PREFIX_SSD_CAP=1" in arms["bgcap"], arms["bgcap"]
    for a, env in arms.items():
        if a != "bgcap":
            assert "TILERL_COLD_PREFIX_SSD_CAP" not in env, (a, env)
    # bgcap differs from bg2 by exactly the cap.
    a, b = arms["bgcap"].split(), arms["bg2"].split()
    assert set(a) - set(b) == {"TILERL_COLD_PREFIX_SSD_CAP=1"}, (a, b)
    assert set(b) - set(a) == set(), (a, b)


def test_reclaim_sampling_runs_only_on_the_arms_that_carry_the_question():
    """Without the cap the shared spill is unbounded and never truncates, so a
    sample of it can only read a plateau -- and because the arm WAITS on the
    sampler, every other arm paid its full duration for a non-result. The two arms
    that carry the question are bgcap (cap: truncation observable) and bg2 (same bg
    config without the cap: the plateau is its control). Sampling bgcap alone would
    state a shrink with nothing to compare it against."""
    src = SRC.read_text()
    assert 'case "$name" in bgcap|bg2) reclaim_on=1 ;; esac' in src, "the gate is not the pair"
    assert "if [ \"$reclaim_on\" = 1 ]" in src, "the gate result is not used"
    # The pair really is cap-vs-no-cap, which is what makes bg2 the control.
    arms = _arm_envs()
    assert "TILERL_COLD_PREFIX_SSD_CAP=1" in arms["bgcap"]
    assert "TILERL_COLD_PREFIX_SSD_CAP" not in arms["bg2"]
    assert set(arms["bgcap"].split()) - set(arms["bg2"].split()) == {
        "TILERL_COLD_PREFIX_SSD_CAP=1"}
    # ... and no OTHER arm samples, so the remaining five do not pay the wait.
    assert sum("TILERL_COLD_PREFIX_SSD_CAP=1" in e for e in arms.values()) == 1


def test_reclaim_span_outlives_the_first_release_it_watches():
    """The sampler is the arm's clock: run_arm waits on it before stopping the
    serve. If its span ends before rep0's first release, it samples the plateau and
    the release it exists to catch is never in its rows -- and the arm still pays
    the full span.

    Arithmetic, from measured numbers: one 32k cold fill prompt ~156 s, the probe's
    --fill-n default 5 (the harness does not pass it), and the warm request is
    itself a 32k prompt. A 60x10 span (590 s) is short of that; the shipped default
    must clear it."""
    import re

    src = SRC.read_text()
    samples = int(re.search(r"RECLAIM_SAMPLES=\$\{RECLAIM_SAMPLES:-(\d+)\}", src).group(1))
    interval = int(re.search(r"RECLAIM_INTERVAL_S=\$\{RECLAIM_INTERVAL_S:-(\d+)\}", src).group(1))
    span = (samples - 1) * interval
    fill_s, fill_n = 156, 5          # measured fill; the probe's default --fill-n
    first_release = fill_n * fill_s + fill_s
    assert span >= first_release, (
        f"sampler span {span}s ends before rep0's first release ~{first_release}s")
    # The old default cost every arm ~40 min, including the five that cannot shrink.
    assert span <= 40 * 60, f"{span}s is the old unbounded wait back again"


def test_reclaim_span_matches_the_documented_arithmetic():
    """The header states the constraint and the numbers it was derived from. Keep
    the two in step: a default changed without the comment is how the coupling
    gets silently broken. Matched case-insensitively -- the assertion is about the
    fact being stated, not about the capitalisation."""
    src = SRC.read_text().lower()
    assert "936 s" in src, "the header no longer states rep0's first release"
    assert "90 x 15" in src, "the header no longer states the shipped span"
    assert "coupled" in src, "the --fill-n coupling is not stated"


def test_reclaim_samples_the_shared_spill_not_the_private_one():
    """The #740 trailing truncation reclaims `<cold-ssd-path>.prefix.bin`
    (kv_tiers._shared_ssd_path). Sampling the private `$COLD_SSD` measured a file
    the effect does not touch, so the reading could only ever be a plateau."""
    src = SRC.read_text()
    assert "${COLD_SSD%.bin}.prefix.bin" in src, src[:200]
    assert '--spill-path "$shared_spill"' in src
    # ... and the sampler is NOT gated on that file existing. The shared spill is
    # created by this window's own first publish, so an existence gate would skip
    # sampling on exactly the arm that creates it; the sampler reads a missing
    # path as size 0, which is what the first rows of a real run look like.
    assert '[ -f "$shared_spill" ]' not in src, "the sampler is gated on existence"


def test_each_arm_gets_its_own_log():
    """steady_filter and the probe read from offset 0, and the supervisor only
    truncates a log already over LOG_CAP at boot -- so a shared fixed path made
    arm N's statistic cover arms 1..N-1 too."""
    src = SRC.read_text()
    assert "SERVE_LOG=$arm_log" in src, "the serve is not given a per-arm log"
    assert '--log "$arm_log"' in src, "the probe does not read the per-arm log"
    assert 'steady_filter.py --log "$arm_log"' in src
    # ... and no reader still points at the shared path
    assert '--log "$LOG"' not in src, "a reader still uses the shared log"


def test_the_steady_filter_is_windowed_to_each_reps_warm_span():
    """A per-arm log is not enough: the supervisor's warmup (dense 7000 + sparse
    9000, 8-token decodes) and each rep's cold FILL write short-context decode
    ticks that PASS the standard set, so reading the whole file reports warmup and
    fill in the steady median.

    A single start offset is not enough either, and this is the subtlety: the probe
    records `log_byte_offset` AFTER the fill and BEFORE the warm POST (a warm
    START), so `[off_i, off_{i+1})` contains warm_i AND fill_{i+1} -- the very ticks
    the windowing claims to drop. Every span must end at that rep's own
    `log_byte_end`.

    The first version of this test shipped the bug: it built ONE `--window off` to
    EOF and only grepped the multi-rep wiring, so the single-rep construction it
    executed was correct while the multi-rep one the harness builds was not. This
    test drives the SHIPPED span extractor over a multi-rep arm.json laid out the
    way the probe records boundaries, and asserts on the filter's output -- the
    only construction that can see an off-by-one-phase span."""
    import json
    import subprocess
    import tempfile

    src = SRC.read_text()
    assert "log_byte_end" in src, "the harness does not read the rep END offsets"
    # No bounded window (missing arm.json, or one predating log_byte_end) means no
    # standard-set figure: the filter must not be run unwindowed or open-ended,
    # either of which reports warmup and a later rep's fill as steady.
    assert "steady.json NOT written" in src, "the no-span path degrades silently"
    assert "raise SystemExit(3)" in src, "a pre-fix arm.json is not refused"
    # `set -u` + an EMPTY array: `"${win_args[@]}"` is an unbound-variable abort on
    # bash 3.2 (macOS's /bin/bash, which is what starts the window from the laptop),
    # and the empty case is reachable -- it is the no-span branch. The guarded
    # expansion is what makes that branch survive there.
    #
    # The negative direction is NOT asserted: bash 5 (the ubuntu-latest row) made an
    # unguarded empty expansion a rc-0 no-op, so "it must fail" is a version
    # artifact rather than the contract. That assertion is what failed CI here -- it
    # was green on macOS bash 3.2 and red on Ubuntu bash 5. Removing the guard is
    # still caught, by the version-independent textual assert above; what this runs
    # is the positive contract, that the guarded form expands correctly on the
    # interpreter running the gate.
    assert '${win_args[@]+"${win_args[@]}"}' in src, "unguarded empty-array expansion"
    guarded = ("set -u\n"
               'A=()\n'
               'f() { printf "%s\\n" "$*"; }\n'
               'f ${A[@]+"${A[@]}"}\n'
               'B=(--window 5)\n'
               'f ${B[@]+"${B[@]}"}\n')
    r = subprocess.run(["bash", "-c", guarded], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["", "--window 5"], r.stdout

    # The exact multi-rep log: warmup, fill0, warm0, fill1, warm1a, warm1b. fill0 is
    # 178 ms and fill1 is 400 ms so one fill would land in the body and the other in
    # the tail -- either way the two headline quantities move.
    def tick(n, total, model):
        return (f"[step-timing] tick {n} total={total}ms dec=1 pre=0 "
                f"model={model}ms sample=3ms path=eager sparse=1")
    lines = [tick(1, 175, 155),   # supervisor warmup
             tick(2, 178, 158),   # fill0
             tick(3, 176, 156),   # warm0
             tick(4, 400, 380),   # fill1  <- the tick the old construction admitted
             tick(5, 177, 157),   # warm1a
             tick(6, 179, 159)]   # warm1b
    d = pathlib.Path(tempfile.mkdtemp(prefix="cwspan."))
    (d / "serve.log").write_text("\n".join(lines) + "\n")
    # Boundaries the way the probe records them: start = after the rep's fill,
    # end = after the rep's warm returns.
    def past(i):
        return sum(len(x) + 1 for x in lines[: i + 1])
    spans = {"rep0": (past(1), past(2)), "rep1": (past(3), past(5))}
    (d / "arm.json").write_text(json.dumps({
        "reps": [{"ticks": {"log_byte_offset": s, "log_byte_end": e}}
                 for s, e in spans.values()]}))

    # Drive the SHIPPED extractor, not a re-implementation of it: pull the inline
    # python block out of the harness source and run it as the harness does, with
    # `$dir` (and its arm.json) supplied. Sliced between the two markers that
    # bracket it, so a reworded comment above cannot silently empty this.
    start = src.index("spans=$(") + len("spans=$(")
    end = src.index("' \"$dir/arm.json\"", start)
    py = src[start:end]
    assert "log_byte_end" in py, py[:200]
    script = ('PYTHON=python3\ndir="$1"\nspans=$(' + py
              + "' \"$dir/arm.json\" 2>\"$dir/spans.note\")\n"
              # The block assigns; the caller is what reads $spans. Echo it so this
              # test observes the value the harness would go on to pass as --window.
              'printf "%s\\n" "$spans"')
    r = subprocess.run(["bash", "-c", script, "_", str(d)], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, (r.returncode, r.stderr)
    got = r.stdout.split()
    want = [f"{s}:{e}" for s, e in spans.values()]
    assert got == want, (got, want)

    # ... and that list, fed to the filter, keeps the warm ticks only. This is the
    # assertion the shipped test could not make.
    args = []
    for s in got:
        args += ["--window", s]
    r = subprocess.run(["python3", str(REPO / "scripts" / "steady_filter.py"),
                        "--log", str(d / "serve.log"), *args],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-400:]
    rec = json.loads(r.stdout)
    assert rec["steady_n"] == 3, rec            # warm0, warm1a, warm1b
    assert rec["ticks_total"] == 3, rec
    assert rec["tail_n"] == 0, rec
    assert rec["steady_p50_ms"] == 177.0, rec
    assert rec["windows"] == [list(spans["rep0"]), list(spans["rep1"])], rec

    # The construction the harness used to build -- [off_i, off_{i+1}), last to EOF
    # -- on the same file. It reports fill1 (400 ms) as a tail tick, which is the
    # live defect: a whole window's tail figures would have carried a fill.
    old = [f"{spans['rep0'][0]}:{spans['rep1'][0]}", str(spans["rep1"][0])]
    args = []
    for s in old:
        args += ["--window", s]
    r = subprocess.run(["python3", str(REPO / "scripts" / "steady_filter.py"),
                        "--log", str(d / "serve.log"), *args],
                       capture_output=True, text=True, timeout=60)
    bad_rec = json.loads(r.stdout)
    assert bad_rec["tail_n"] == 1 and bad_rec["tail_max_ms"] == 400, bad_rec
    assert bad_rec["ticks_total"] > rec["ticks_total"], (bad_rec, rec)


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
    # EXACT. `in (179, 181)` was written here first and is exactly the hole rev
    # found: it accepts both the true median (179) and nearest-rank (181), which
    # is the convention mismatch this script exists to prevent.
    assert rec["steady_p50_ms"] == 179.0, rec
    assert rec["excluded_path_graph_n"] == 1, rec


def test_steady_median_matches_the_tree_convention():
    """The number must equal what the tree already calls a median, on the sample
    sizes a warm window yields. `probe_draft_window_sweep` reports `tick_ms_med`
    with `statistics.median` and `probe_device_artifacts_crosscheck` recomputes
    arm medians the same way, so a headroom arm is only comparable to a sweep arm
    if this agrees with both -- on EVEN n, where nearest-rank and `int(q*n)`
    differ from it."""
    import importlib.util
    import statistics

    spec = importlib.util.spec_from_file_location("sf", REPO / "scripts" / "steady_filter.py")
    sf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sf)
    for xs in ([176, 180], [176, 178, 180, 1000], [176, 178, 180, 182]):
        assert sf.median(xs) == statistics.median(xs), xs
    # ... and it must NOT be either of the two conventions it is distinguished from.
    assert sf.median([176, 180]) != 176, "median fell back to nearest-rank"
    assert sf.median([176, 180]) != 180, "median fell back to int(q*n)"
    assert sf.median([]) is None
    # A percentile is nearest-rank, deliberately: that is the tree's pNN rule,
    # and it must agree with BOTH tree helpers that use it.
    assert sf.pct([176, 180], 0.5) == 176, sf.pct([176, 180], 0.5)
    import importlib.util as _u
    for name, path in (("probe_draft_window_sweep", "probe_draft_window_sweep.py"),
                       ("probe_headroom_coldtail", "probe_headroom_coldtail.py")):
        sp = _u.spec_from_file_location(name, REPO / "scripts" / path)
        mod = _u.module_from_spec(sp)
        sp.loader.exec_module(mod)
        tree_pct = getattr(mod, "_pct", None) or getattr(mod, "pct")
        for xs in ([176, 180], [176, 178, 180, 1000], [100, 200, 300, 400, 500]):
            for q in (0.1, 0.5, 0.9):
                assert sf.pct(xs, q) == tree_pct(xs, q), (name, xs, q)


def test_follower_outcomes_have_distinct_exit_codes():
    """MISMATCH and NO-PREFIX-HIT are different findings, so the shell rc must not
    collapse them: a caller branching on rc alone has to be able to tell a
    correctness bug from a store that did not serve."""
    src = SRC.read_text()
    for code, why in ((3, "MISMATCH"), (4, "NO-PREFIX-HIT"), (5, "block leak"), (6, "cancel")):
        assert re.search(rf"raise SystemExit\(\{{.*?{why}.*?\}}\[", src) or str(code) in src, code
    assert '"MISMATCH": 3' in src and '"NO-PREFIX-HIT": 4' in src
    assert "[ \"$rc_c\" != 0 ] && return 6" in src
    # The doc and the script must state the same table.
    doc = (REPO / "docs" / "run-close-window-v100.md").read_text()
    for code in ("| 3 |", "| 4 |", "| 5 |", "| 6 |"):
        assert code in doc, code


def test_the_shared_fuse_blocks_the_next_arm_and_per_arm_fuse_does_not():
    """The supervisor's crash-burst fuse lives in `$ROOT/.servehybridsse.fuse` and,
    once RESTART_FUSE_MAX restarts land inside RESTART_FUSE_WINDOW_S, it stays down
    and exits 2 WITHOUT starting python. `$ROOT` is shared by every arm, so a
    crash-looping arm left a tripped fuse that stopped the NEXT arm's serve from
    ever running -- which this harness reports as that arm's result, since the
    health gate can only say "never became ready".

    RED first, then GREEN, both executed against the real launcher rather than
    grepped:

      RED   -- old behaviour (one shared fuse, nothing cleared): a pre-tripped fuse
               makes the boot exit 2 and the stub python is never invoked.
      GREEN -- per-arm fuse, cleared before boot: that arm starts, and a SECOND arm
               with its own path starts too even though the first arm's fuse file
               now holds entries.
    """
    import os as _os
    import pathlib as _p
    import subprocess as _sp
    import tempfile as _tf

    from _flock_shim import flock_path

    src = SRC.read_text()
    assert "SERVE_FUSE_STATE=$arm_fuse" in src, "the arm is not given its own fuse file"
    assert 'rm -f "$arm_fuse"' in src, "the arm's fuse file is not cleared before boot"
    assert 'rm -f "$ROOT/.servehybridsse.fuse"' not in src, "the prod fuse must not be deleted"

    with flock_path() as path:
        d = _p.Path(_tf.mkdtemp(prefix="fusegate."))
        (d / "tilerl-v100-sse").mkdir()
        (d / "venv70/bin").mkdir(parents=True)
        (d / "models").mkdir()
        (d / "mmlu-assets").mkdir()
        boots = d / "boots"
        stub = d / "venv70/bin/python"
        stub.write_text(f'#!/bin/bash\necho x >> {boots}\nexit 7\n')
        stub.chmod(0o755)
        launcher = str((SRC.parent / "serve_hybrid_v100.sh").resolve())

        def boot(fuse_path, tag):
            env = dict(_os.environ)
            env.update({"SERVE_ROOT": str(d), "SERVE_REPO": str(d / "tilerl-v100-sse"),
                        "SERVE_PYTHON": str(stub), "SERVE_LOG": str(d / f"{tag}.log"),
                        "SERVE_LOCK": str(d / f"{tag}.lock"),
                        "SERVE_FUSE_STATE": str(fuse_path),
                        "MAX_RESTARTS": "0", "PATH": path})
            return _sp.run(["bash", launcher], capture_output=True, text=True,
                           timeout=120, env=env, cwd=str(d / "tilerl-v100-sse"))

        # ---- RED: the OLD wiring, i.e. one shared fuse that is never cleared.
        # A pre-tripped shared fuse (>= RESTART_FUSE_MAX entries stamped "now") is
        # exactly the state a crash-looping arm leaves behind.
        shared = d / "shared.fuse"
        shared.write_text("".join(f"{int(__import__('time').time())}\n" for _ in range(8)))
        before = boots.read_text().count("x") if boots.exists() else 0
        r = boot(shared, "red")
        after = boots.read_text().count("x") if boots.exists() else 0
        assert r.returncode == 2, f"expected the fuse to stay down, got rc={r.returncode}"
        assert after == before, "python ran despite a tripped fuse; RED is vacuous"
        log = (d / "red.log").read_text()
        assert "FUSE:" in log, log[-300:]

        # ---- GREEN: per-arm fuse paths, cleared before each boot (what the
        # harness does now). Arm A starts even with A's own stale fuse present
        # because the harness clears it; arm B starts with a different path.
        for tag in ("armA", "armB"):
            arm_fuse = d / f"{tag}.fuse"
            arm_fuse.write_text("".join(f"{int(__import__('time').time())}\n" for _ in range(8)))
            _sp.run(["bash", "-c", f'rm -f "{arm_fuse}"'])   # the shipped rm -f
            before = boots.read_text().count("x") if boots.exists() else 0
            r = boot(arm_fuse, tag)
            after = boots.read_text().count("x") if boots.exists() else 0
            assert r.returncode != 2, f"{tag}: fuse blocked a cleared per-arm fuse"
            assert after > before, f"{tag}: python never started under its own fuse"


def test_restore_uses_the_production_fuse_read_only():
    """`--restore-only` targets the shipped serve, so it must inherit the PRODUCTION
    fuse under $ROOT (no SERVE_FUSE_STATE export) and must never delete it -- arming
    the prod fuse again is an operator decision, same as when it trips on its own.
    It warns instead, and the warning has to name the file."""
    src = SRC.read_text()
    body = src.split("restore() {", 1)[1].split("\n}", 1)[0]
    # Assertions are on CODE, not prose: a comment explaining why the fuse path is
    # not overridden must not read as an override. Drop comment lines and the
    # heredoc python block (which only prints), then assert on what executes.
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = code.split("<<'PY'", 1)[0]
    assert "export SERVE_FUSE_STATE" not in code, "restore overrides the fuse path"
    for bad in ("rm -f", "unlink"):
        assert bad not in code, f"restore mutates the prod fuse: {bad}"
    assert "$ROOT/.servehybridsse.fuse" in code, "restore does not name the prod fuse"
    assert "WARNING" in code, "restore does not warn when the prod fuse is tripped"
