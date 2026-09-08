"""The run ledger: ids are a function of the inputs, a finished run is not
rerun, gates are data in the manifest and an exit code in the CLI."""

import contextlib
import json

import pytest

from tilerl.cli import _EVAL_CONCURRENCY, _build_parser, cmd_ledger, cmd_train
from tilerl.kv_cache import BLOCK_TOKENS
from tilerl.ledger import (
    format_run,
    gates_pass,
    lineage,
    list_runs,
    new_manifest,
    now,
    read_manifest,
    run_id,
    write_manifest,
)


def test_run_id_is_canonical():
    assert run_id({"a": 1, "b": [2, 3]}) == run_id({"b": [2, 3], "a": 1})
    assert run_id({"a": 1}) != run_id({"a": 2})
    assert len(run_id({})) == 12


def test_manifest_round_trip_and_lineage(tmp_path):
    parent = new_manifest("train", {"x": 1})
    child = new_manifest("eval", {"x": 2}, parents=[parent["id"]])
    child["gates"] = [{"name": "g", "value": 1, "threshold": 0, "passed": True}]
    for m in (parent, child):
        write_manifest(tmp_path, m)
    assert read_manifest(tmp_path, child["id"]) == child and gates_pass(child)
    # `finished` has to be set for a verdict to mean anything: gates are written by
    # `_finish`, so an unfinished manifest carries none and `gates_pass([])` is True.
    # This line asserted `pass` on a manifest that never finished, which is the defect
    # a killed run hit -- `e069c8ff28b7 train running pass` on cpu.
    assert format_run(child).split()[3] == "killed"
    child["finished"] = now()
    assert format_run(child).split()[3] == "pass"
    # The fourth verdict: finished with no gates DEFINED. Distinct from `skip`, which in
    # this tree means a gate existed and was suppressed, and from `pass`, which is what
    # `gates_pass([])` returned before -- a judgement over zero checks. `tilerl merge` is
    # the producer; `test_merge.py` asserts it end to end.
    gateless = dict(child, gates=[])
    assert format_run(gateless).split()[3] == "none", format_run(gateless)
    assert gates_pass(gateless), "gates_pass is unchanged, so exit codes are unchanged"
    assert [m["id"] for m in lineage(tmp_path, child["id"])] == [child["id"], parent["id"]]
    assert {m["id"] for m in list_runs(tmp_path)} == {parent["id"], child["id"]}
    assert read_manifest(tmp_path, "missing") is None


def _train(argv):
    """cmd_train exits non-zero on a failed gate; return the code."""
    try:
        cmd_train(_build_parser().parse_args(["train", *argv]))
    except SystemExit as e:
        return e.code
    return 0


@pytest.mark.parametrize("mode", ["--rl", "--opd"])
def test_zero_steps_writes_eval_manifest(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "eval.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    assert _train([mode, "--model", "tiny", "--steps", "0", "--data", str(data),
                   "--eval-gsm8k", str(data), "--eval-max-new-tokens", "4"]) == 0
    (m,) = list_runs(tmp_path / "runs")
    assert m["finished"] and isinstance(m["metrics"]["gsm8k_after"], int)
    assert m["metrics"]["gsm8k_after_tokens"] > 0
    assert not {"reward_first", "reward_last", "ce_last", "secs_per_step_median",
                "secs_total", "tied_group_fraction", "tokens_first", "tokens_last",
                "rollout_secs", "backward_secs", "optimizer_secs"} & m["metrics"].keys()
    assert all(g["skipped"] and g["passed"] is None for g in m["gates"])
    assert gates_pass(m)
    assert format_run(m).split()[3] == "skip"


def test_train_cli_writes_manifest_and_is_idempotent(tmp_path, monkeypatch, capsys):
    """`tilerl train --rl --data --eval-gsm8k` on the tiny model: the plumbing
    the 27B run uses — JSONL prompts, ChatML, exact-match reward, GSM8K greedy
    eval before and after — and a manifest a second identical call returns
    from instead of retraining."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n{"prompt": "2+2?", "answer": "4"}\n')
    # --allow-short-rollouts: max_new_tokens 4 is deliberately below any real
    # completion here, which is exactly what the length guard refuses.
    argv = ["--rl", "--data", str(data), "--eval-gsm8k", str(data), "--steps", "2",
            "--group", "2", "--max-new-tokens", "4", "--lora-rank", "4",
            "--allow-short-rollouts"]
    code = _train(argv)
    (m,) = list_runs(tmp_path / "runs")
    assert [g["name"] for g in m["gates"]] == [
        "rollouts_within_cap", "reward_rises", "mmlu_holds", "gsm8k_improves",
        "groups_untied", "ce_falls"]
    # ce_falls carries no ce_first on the RL path, so it is recorded as not measured
    # rather than passed -- see test_ce_falls_is_not_measured_on_the_rl_path.
    assert m["metrics"].get("ce_first") is None
    ce = next(g for g in m["gates"] if g["name"] == "ce_falls")
    assert ce["skipped"] is True and ce["passed"] is None, ce
    assert code == (0 if gates_pass(m) else 1)
    assert isinstance(m["metrics"]["gsm8k_before"], int)
    assert isinstance(m["metrics"]["gsm8k_after"], int)
    assert m["metrics"]["mmlu_before"] is None and m["inputs"]["source"] == "tiny"

    capsys.readouterr()
    assert _train(argv + ["--json"]) == code
    again = json.loads(capsys.readouterr().out)
    assert again["finished"] == m["finished"], "a finished run was retrained"

    cmd_ledger(_build_parser().parse_args(["ledger", "--json"]))
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == [m["id"]]


def test_the_manifest_records_the_engine_config_the_wall_clock_depends_on(tmp_path, monkeypatch):
    """A run's wall clock cannot be compared against another run's without these.

    Six card sessions on the 2.6x rollout tick recovered their two pool sizes only
    because the probe script logged its own flags; the manifest recorded none of the
    bundle. Per-forward device time moves with it, so P5 (against verl+sglang) is
    exactly the comparison a record without it cannot support.

    Two things are asserted, not one. The keys must be present AND `blocks` must
    track the pool -- a key list alone goes green over a hardcoded dict, and
    `--max-new-tokens` is what sizes the training pool
    (`num_blocks = ceil(ctx/16)*group + 8`), so two runs differing only there must
    report different pools. `slots` and `max_batch` are asserted against --group
    rather than a literal 8: they used to BE literal 8 while --group was settable,
    which made `--group 16` queue into two waves of 8 with nothing raising.

    4000, not 200: `ctx` has a 1024 floor (`cli.py:568`), and at 200 both arms land
    on it and report 520 blocks each. Written with 200 first, and the pair-assert is
    what caught it -- a key-presence check would have passed on two identical pools.

    The pool must cover the eval arms, which submit into this same engine. #320 asserted
    that on the ROW axis (`max(group, _EVAL_CONCURRENCY)`) and was wrong: only `num_slots`
    rows hold blocks at once, and the axis that was actually short is per-row LENGTH, since
    `ctx` used the rollout's cap while the eval arms run at `--eval-max-new-tokens`
    (default 2048). Both arms of that pair pass with the row assertion deleted, so the
    length arms below are what carry this test now.

    `--eval-max-new-tokens` is passed EXPLICITLY in the length arms and left DEFAULT in the
    last one. Every arm in the #320 version overrode it, which is why a green suite shipped
    a pool that exhausts under `tilerl train --rl` as a user runs it: branch coverage and
    configuration coverage are different things, and that suite had the first.
    """
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    seen = {}
    # (2, 4000) is the only arm above the 1024 `ctx` floor on the rollout's cap, so it is
    # what moves if that floor changes; the others sit on it or are raised by the eval cap.
    for group, new, ecap in ((2, 4, 4), (2, 4000, 4), (16, 4, 4), (2, 4, 2048)):
        _train(["--rl", "--data", str(data), "--steps", "0", "--group", str(group),
                "--max-new-tokens", str(new), "--lora-rank", "4",
                "--eval-max-new-tokens", str(ecap)])
        (m,) = [r for r in list_runs(tmp_path / "runs")
                if r["inputs"]["max_new_tokens"] == new and r["inputs"]["group"] == group
                and r["inputs"]["eval_max_new_tokens"] == ecap]
        assert m["engine"].keys() == {
            "blocks", "slots", "max_batch", "max_total_tokens",
            "max_num_batched_tokens", "decode_graph", "prefix_store", "spec_width"}
        # Tracks --group, not a literal: the training engine is sized from it, so a
        # frozen 8 here would pass while production computed something else.
        assert m["engine"]["slots"] == group == m["engine"]["max_batch"]
        assert m["engine"]["prefix_store"] == "NoPrefixStore"
        # Every in-flight row's whole sequence, priced per consumer and maxed -- the same
        # shape as production, not a re-derivation of it, so it moves with the defaults.
        # min(group, 8) on the eval side because a submit past num_slots queues.
        need = max(group * -(-(new + 64 + 8) // BLOCK_TOKENS),
                   min(group, _EVAL_CONCURRENCY) * -(-(ecap + 64 + 8) // BLOCK_TOKENS))
        assert m["engine"]["blocks"] >= need, (group, new, ecap, m["engine"]["blocks"], need)
        seen[group, new, ecap] = m["engine"]["blocks"]
        # Not in `inputs`: the id hashes inputs, so a pool field there would make
        # every pool change a new run instead of a rerun.
        assert "blocks" not in m["inputs"]
    assert seen[2, 4000, 4] > seen[2, 4, 4], f"blocks did not track the context: {seen}"
    # Same group, same rollout cap, longer EVAL cap: the axis #320 left out.
    assert seen[2, 4, 2048] > seen[2, 4, 4], f"blocks did not track the eval cap: {seen}"
    # Same ctx, wider group: the pool must still grow with the rows.
    assert seen[16, 4, 4] > seen[2, 4, 4], f"blocks did not track the group: {seen}"


def test_the_eval_curve_records_the_step_a_score_was_reached_at(tmp_path, monkeypatch):
    """`time_to_score = steps_to_score x seconds_per_step` needs the STEP, and
    gsm8k_before/after cannot say which step a score was crossed at.

    Three assertions, because the triple is only useful whole. `step` is the
    numerator's operand, `secs` is the product, and `score` is what lets the
    threshold live at the reading end -- the ledger records scores and never
    decides which one counts.

    `secs` is asserted MONOTONE and equal to `secs_total` at the last point, not
    merely present. The first version of this accumulated into a local named
    `elapsed`, which the timings loop 8 lines below rebinds every step, so the
    curve reported 0.143 s at step 4 against 0.148 at step 2 -- a cumulative
    figure going down, which a presence check passes.
    """
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n{"prompt": "2+2?", "answer": "4"}\n')
    argv = ["--rl", "--data", str(data), "--eval-gsm8k", str(data), "--steps", "4",
            "--group", "2", "--max-new-tokens", "4", "--lora-rank", "4",
            "--allow-short-rollouts", "--eval-max-new-tokens", "4"]
    _train([*argv, "--eval-every", "2", "--eval-curve-n", "2"])
    (m,) = list_runs(tmp_path / "runs")
    curve = m["eval_curve"]
    assert curve["every"] == 2 and curve["n"] == 2
    assert [p["step"] for p in curve["points"]] == [2, 4], curve
    secs = [p["secs"] for p in curve["points"]]
    assert secs == sorted(secs), f"cumulative seconds are not monotone: {secs}"
    assert secs[-1] == pytest.approx(m["metrics"]["secs_total"], abs=0.01), (
        f"the last point's {secs[-1]} s should be the run's own "
        f"{m['metrics']['secs_total']} s")
    for p in curve["points"]:
        assert 0.0 <= p["score"] <= 1.0 and p["correct"] <= p["total"] == 2
        # Each point prices its own scoring, so "keep the eval under 5% of a step" is
        # checkable after a run instead of estimated before one. Asserted > 0 rather
        # than merely present: a zero would mean the clock never ran.
        assert p["eval_secs"] > 0.0, p
        # `mean_len` and `at_cap` travel with the score because a score alone cannot say
        # whether it is the policy's or the cap's: 2026-09-04 shipped 39.0% that was the
        # latter (mean completion 238.7 against a 256 cap, ~82.5% uncapped). Asserted as
        # a RANGE, not presence -- mean_len must lie in (0, cap] and at_cap in [0, total].
        assert 0 < p["mean_len"] <= 4, p          # --eval-max-new-tokens 4 in this argv
        assert 0 <= p["at_cap"] <= p["total"], p
    # Exactly one point carries the JIT, and it is the first: on a real 27B run
    # tilerl-0a measured 2.801 s against 0.500 s at identical n, so eval_secs is not
    # comparable across that boundary. The flag is a field and not a comment because a
    # reader of the manifest cannot tell which point was first.
    assert [p["jit"] for p in curve["points"]] == [True, False], curve

    # The per-problem rows reach disk, one file per point, because comparing the curve's
    # points to each other is PAIRED -- every point scores the same `curve_rows`. Unpaired,
    # adjacent points carry a 1.90 pt difference SE at n=500; paired at 5% discordant it is
    # 1.00 pt, and "has the score stopped rising" is a question about a difference smaller
    # than either arm. P1 fell back to the unpaired interval for want of exactly these rows.
    #
    # Asserted on CONTENT, not on the file existing: `i` must identify the row so two points
    # can be joined, and the set of `i` must be identical across points or the join is over
    # different problems.
    seen_i = []
    for p in curve["points"]:
        f = tmp_path / "runs" / m["id"] / f"eval-curve-{p['step']}.jsonl"
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        assert len(rows) == p["total"], (f, len(rows), p)
        assert all({"i", "correct", "tokens"} <= r.keys() for r in rows), rows[:1]
        seen_i.append(sorted(r["i"] for r in rows))
    assert seen_i[0] == seen_i[1], f"points scored different problems: {seen_i}"

    # Off by default, so no existing invocation changes shape.
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs2"))
    _train(argv)
    (off,) = list_runs(tmp_path / "runs2")
    assert "eval_curve" not in off


def test_time_to_score_returns_the_crossing_point_and_never_interpolates():
    """`time_to_score = steps_to_score x seconds_per_step` read off a synthetic curve.

    Synthetic, not a training run: the reader is arithmetic over recorded points and
    a real run would make the SCORES the variable under test instead of the reading.
    Three distinct answers, because collapsing any two of them loses information a
    reader needs:

    * reached -- the point that crossed, plus the interval `(after_step, step]` it was
      crossed in. NOT an interpolated step: the target sits between two scoring points
      and only the right end was measured, and this figure is the project's headline
      metric. A fitted headline is exactly what the curve exists to prevent.
    * instrumented but never reached -- `reached=False` with the best score, so nobody
      reads the last point as if it were the target.
    * no curve at all -- None. "Not instrumented" and "instrumented and fell short"
      are different facts about a run and a single falsy answer conflates them.
    """
    from tilerl.ledger import time_to_score

    curve = {"n": 20, "every": 10, "points": [
        {"step": 10, "correct": 1, "total": 20, "score": 0.05, "secs": 100.0},
        {"step": 20, "correct": 4, "total": 20, "score": 0.20, "secs": 210.0},
        {"step": 30, "correct": 9, "total": 20, "score": 0.45, "secs": 320.0},
    ]}
    m = {"eval_curve": curve}

    hit = time_to_score(m, 0.30)
    assert hit["reached"] and hit["step"] == 30 and hit["after_step"] == 20, hit
    assert hit["secs"] == 320.0 and hit["score"] == 0.45
    # The interval is the whole point: 0.30 was crossed somewhere in (20, 30] and the
    # reader must not be handed a step nobody scored at.
    assert hit["step"] != 24 and 20 < hit["step"] <= 30

    # An exact hit at the first point still reports after_step 0, not a missing key.
    first = time_to_score(m, 0.05)
    assert first["step"] == 10 and first["after_step"] == 0

    miss = time_to_score(m, 0.90)
    assert miss["reached"] is False and miss["steps_run"] == 30
    assert miss["best"] == 0.45 and miss["secs"] == 320.0

    assert time_to_score({}, 0.1) is None, "no curve is None, not a miss"
    assert time_to_score({"eval_curve": {"points": []}}, 0.1) is None

    # The subset's width travels with the answer, both when it reached and when it did
    # not, and it is the POINT's own rate rather than p=0.5's worst case. Asserting the
    # number, not just the key: the whole point is that a reader sees how wide it is.
    # (tilerl-0a named the resolution.)
    assert hit["n"] == 20 and hit["total"] == 20 and hit["correct"] == 9
    assert hit["se_pt"] == 11.12, hit          # 9/20 = 0.45
    assert miss["n"] == 20 and miss["se_pt"] == 11.12
    wide = time_to_score({"eval_curve": {"n": 500, "points": [
        {"step": 10, "correct": 300, "total": 500, "score": 0.60, "secs": 9.0}]}}, 0.55)
    assert wide["se_pt"] == 2.19, wide         # 300/500 = 0.60
    # 2.19 < 5.0 so the reader stays silent there, and 11.12 >= 5.0 so it warns.
    from tilerl.cli import _se_note
    assert _se_note(wide) == "" and "sampling-limited" in _se_note(hit)

    # The fixtures above all sit near p=0.5, where `p(1-p)` is flat -- the old
    # `0.25/n` hardcode is right to 2% there, so those assertions passed while the
    # width was computed at the worst case rather than the point's. This one is at the
    # product's real operating rate, where the two answers differ 2x. Negative control:
    # putting `0.25` back gives 2.24 here, and this assertion is what catches it.
    real = time_to_score({"eval_curve": {"n": 500, "points": [
        {"step": 25, "correct": 466, "total": 500, "score": 0.932, "secs": 537.4}]}}, 0.91)
    assert real["se_pt"] == 1.13, (real, "466/500 = 0.932, not p=0.5's 2.24")
    assert _se_note(real) == "", "1.13 pt resolves P1's +5 pt; warning must stay silent"

    # A transient crossing is reported AS a crossing, with `held` False and the step it
    # fell back at. Not suppressed: requiring every later point to stay above would turn
    # one noisy dip into "never reached" -- at target 0.90 a 0.89 point is 0.01 below
    # against an SE of 0.014, well inside noise -- and that is a false negative on a
    # number later runs are priced against. Both facts, and the reader decides.
    dip = time_to_score({"eval_curve": {"n": 500, "points": [
        {"step": 10, "correct": 455, "total": 500, "score": 0.91, "secs": 100.0},
        {"step": 20, "correct": 440, "total": 500, "score": 0.88, "secs": 200.0},
        {"step": 30, "correct": 445, "total": 500, "score": 0.89, "secs": 300.0}]}}, 0.90)
    assert dip["reached"] and dip["step"] == 10, dip
    assert dip["held"] is False and dip["dipped_at"] == 20, dip
    held = time_to_score({"eval_curve": {"n": 500, "points": [
        {"step": 25, "correct": 466, "total": 500, "score": 0.932, "secs": 537.4},
        {"step": 50, "correct": 467, "total": 500, "score": 0.934, "secs": 1038.6}]}}, 0.91)
    assert held["reached"] and held["step"] == 25 and held["held"] is True, held
    assert held["dipped_at"] is None, held

    # `best` and its width come from the BEST point, not the last one: the last can be a
    # lower score, and a width read off it describes a different number than the one
    # printed beside it.
    fell = time_to_score({"eval_curve": {"n": 500, "points": [
        {"step": 10, "correct": 466, "total": 500, "score": 0.932, "secs": 100.0},
        {"step": 20, "correct": 250, "total": 500, "score": 0.50, "secs": 200.0}]}}, 0.99)
    assert fell["reached"] is False and fell["best"] == 0.932, fell
    assert fell["se_pt"] == 1.13, (fell, "the width of 0.932, not of the last point 0.50")
    assert fell["secs"] == 200.0, "secs is the run's cost, which is the LAST point's"


def test_periodic_rollout_guard_stops_at_first_window_crossing(tmp_path, monkeypatch, capsys):
    from contextlib import suppress

    from tilerl import cli, train
    from tilerl.engine import Engine
    from tilerl.ledger import gates_pass, list_runs

    root = tmp_path / "runs"
    monkeypatch.setenv("TILERL_RUNS", str(root))
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt": "1+1?", "answer": "2"}\n')
    lengths = [6, 8, 10, 12, 14, 16, 18, 20, 20, 20]
    sampled = []
    on_disk = []
    identified = []

    def rollout(engine, ids, what):
        # Sampled at the START of each step, so it reads what earlier steps wrote:
        # a single write after the loop leaves this all zeros, and a run killed
        # mid-training keeps nothing. Run 2 died on a SIGTERM at step 45.
        found = [p for p in root.glob("*/rollouts.jsonl")]
        on_disk.append(sum(len(p.read_text().splitlines()) for p in found))
        # And the rows are useless without the run that made them: rollouts.jsonl
        # carries no model, cap, group, lr, seed or commit, and the directory name
        # is a hash of those. Measured before the fix: a SIGTERM at step 6 left 12
        # rows, no manifest, and `tilerl ledger --json` printed [].
        identified.append(len(list_runs(root)))
        n = lengths[len(sampled)]
        sampled.append(n)
        return {i: [4] * n for i in ids}

    # Keep the real GRPO loop and CLI; only generation and the expensive update are stubbed.
    def update(*a, timings, **kw):
        timings.update(backward_secs=0.01, optimizer_secs=0.001)
        return 1.0

    monkeypatch.setattr(train, "_drain", rollout)
    monkeypatch.setattr(train, "rl_step", update)
    argv = ["train", "--rl", "--data", str(data), "--steps", "10", "--group", "2",
            "--max-new-tokens", "20", "--lora-rank", "2"]
    for allow, expected in ((False, 9), (True, 10)):
        sampled.clear()
        on_disk.clear()
        identified.clear()
        # No pending requests in the stubbed drain: submit only supplies unique ids.
        requests = iter(range(20))
        monkeypatch.setattr(Engine, "submit", lambda *a, **kw: next(requests))
        with suppress(SystemExit):
            cli.cmd_train(cli._build_parser().parse_args(
                argv + (["--allow-short-rollouts"] if allow else [])))
        assert len(sampled) == expected, "periodic guard stopped at the wrong step"
        # Every completed step is on disk before the next one starts. The baseline is
        # arm 1's rows, still there when arm 2 runs; the deltas are what this asserts.
        assert [n - on_disk[0] for n in on_disk] == [2 * i for i in range(expected)], (
            f"rollout rows are not written per step: {on_disk}")
        # This run is in the ledger from its first step, not only once it finishes.
        # A run interrupted before `_finish` is otherwise a directory of rows nothing
        # can attribute: `list_runs` globs */manifest.json and would return [].
        assert min(identified) >= 1, (
            f"the run was not in the ledger while it ran: list_runs saw {identified}")
        m = next(m for m in list_runs(root) if m["inputs"]["allow_short_rollouts"] == allow)
        gate = next(g for g in m["gates"] if g["name"] == "rollouts_within_cap")
        assert m["metrics"]["steps_completed"] == expected
        assert (root / m["id"] / m["artifacts"]["adapter"]).is_file()
        if allow:
            assert gate["skipped"] and gate["passed"] is None
        else:
            assert not gate["skipped"] and gate["passed"] is False and not gates_pass(m)
            assert gate["value"] == pytest.approx(17.6)
            assert (gate["threshold"], gate["step"]) == (16.0, 9)
            # The after-arm never ran, so these must read "not measured", not "pass":
            # _finish scores a None value as passed, which is how a stopped run would
            # report mmlu_holds and gsm8k_improves green having measured neither.
            for name in ("mmlu_holds", "gsm8k_improves"):
                g = next(x for x in m["gates"] if x["name"] == name)
                assert g["skipped"] and g["passed"] is None, name
            assert m["metrics"].get("mmlu_after") is None
            assert "step 9" in gate["reason"] and "--max-new-tokens is 20" in gate["reason"]
            assert gate["reason"] in capsys.readouterr().out


def test_sft_writes_a_manifest_and_gates_on_the_loss_falling(tmp_path, monkeypatch, capsys):
    """`tilerl train` without --rl/--opd wrote no manifest at all, so sft-iso-27b
    -- a recipe whose whole purpose is a P3 verdict -- had nowhere to record one.
    The ledger is per-run, not per-algorithm."""
    monkeypatch.setenv("TILERL_RUNS", str(tmp_path / "runs"))
    argv = ["--model", "tiny", "--steps", "4"]
    code = _train(argv)
    (m,) = list_runs(tmp_path / "runs")
    assert m["inputs"]["algo"] == "sft" and m["inputs"]["optim"] == "adafactor"
    assert code == (0 if gates_pass(m) else 1)
    ce = m["metrics"]
    assert ce["ce_first"] is not None and ce["ce_last"] is not None
    assert ce["secs_per_step_median"] is not None
    # The RL gates have no metrics on this path and must pass vacuously.
    for g in m["gates"]:
        if g["name"] != "ce_falls":
            assert g["passed"] and g["value"] is None, g

    capsys.readouterr()
    assert _train(argv + ["--json"]) == code
    again = json.loads(capsys.readouterr().out)
    assert again["finished"] == m["finished"], "a finished SFT run was retrained"


if __name__ == "__main__":  # runnable check
    test_run_id_is_canonical()
    print("ledger: ids OK")


def test_mmlu_score_reports_the_concurrency_it_used():
    """A score whose value depends on concurrency has to carry it.

    concurrency sets B, B sets M = B*W, and M picks the fp4 linear arm across
    the _MGEMV/_MX boundaries -- so two concurrencies can run two kernels on one
    question, and the 27B showed 4 of 1000 answers moving between them. The two
    callers disagreed silently (cli.py 8, scripts/mmlu.py the default 32).

    Gated on the signature rather than end to end: mmlu_accuracy needs the real
    dataset, and what regresses is a caller unpacking two values again.
    """
    import inspect

    from tilerl.eval import mmlu_accuracy

    src = inspect.getsource(mmlu_accuracy)
    assert "concurrency" in src.split("return")[-1], (
        "mmlu_accuracy must return the concurrency it scored at:\n" + src)

    cli = inspect.getsource(__import__("tilerl.cli", fromlist=["_"]))
    call = next(ln for ln in cli.splitlines() if "mmlu_accuracy(" in ln and "import" not in ln)
    assert call.count(",") >= 2 and "conc" in call, f"cli.py drops the concurrency: {call!r}"
    assert '_concurrency"] = conc' in cli, "cli.py must record it in the manifest"


def _gate(name, metrics):
    """Score one gate through `_finish`, the real code path, on a synthetic manifest.

    `_finish` exits non-zero when any gate fails, which is the behaviour under test, so
    the SystemExit is expected rather than an error -- the manifest it wrote is still on
    `m`, and that is what carries the verdict.
    """
    from tilerl.cli import _finish

    m = new_manifest("train", {"steps": 100, "source": "tiny"}, [])
    m["metrics"] = dict(metrics)
    with contextlib.suppress(SystemExit):
        _finish(m, as_json=True)
    return next(g for g in m["gates"] if g["name"] == name)


def test_p1_exit_thresholds_match_the_roadmap(capsys):
    """The roadmap's P1 numbers, encoded in the units each metric is actually written in.

    `gsm8k_{tag}` is a COUNT and `mmlu_{tag}` is a fraction (cli.py:588 vs :575), so the
    same "+5 pt / -2 pt" sentence needs two different encodings. Both arms of each case
    are asserted: the old `after > before` form passes at +1 question of 500, which is
    +0.2 pt against the roadmap's own SE of ~2 pt, so a test that only checked the pass
    case would have gone green on the pre-fix code too.
    """
    base = {"gsm8k_before": 181, "gsm8k_before_total": 500,
            "gsm8k_after_total": 500, "tied_group_fraction": 0.3}

    # +1 of 500 = +0.2 pt: the noise pass this gate accepted before.
    g = _gate("gsm8k_improves", {**base, "gsm8k_after": 182})
    assert g["passed"] is False, f"+0.2 pt must not pass P1: {g}"
    assert g["threshold"] == 181 + 25, g
    capsys.readouterr()

    # +24 of 500 = +4.8 pt, just under; +25 = +5.0 pt, exactly the bar.
    assert _gate("gsm8k_improves", {**base, "gsm8k_after": 205})["passed"] is False
    assert _gate("gsm8k_improves", {**base, "gsm8k_after": 206})["passed"] is True
    capsys.readouterr()

    # The denominator comes off the manifest: 200 questions makes the bar +10.
    g = _gate("gsm8k_improves", {**base, "gsm8k_after": 191,
                                 "gsm8k_after_total": 200, "gsm8k_before_total": 200})
    assert g["threshold"] == 181 + 10 and g["passed"] is True, g
    capsys.readouterr()

    # The `or` fallback, which no other case reaches: a guard stop can leave the
    # after-arm's total unwritten while the before-arm's is on the manifest, and a bar
    # computed from a missing total would be `None` -- a vacuous pass on the gate that
    # matters most. Mutation-driven: dropping the fallback survived every case above.
    g = _gate("gsm8k_improves", {"gsm8k_before": 181, "gsm8k_before_total": 500,
                                 "gsm8k_after": 182, "tied_group_fraction": 0.3})
    assert g["threshold"] == 181 + 25 and g["passed"] is False, g
    capsys.readouterr()

    # MMLU is a fraction and the roadmap allows -2 pt, not -3.
    mm = {**base, "gsm8k_after": 206, "mmlu_before": 0.601}
    assert _gate("mmlu_holds", {**mm, "mmlu_after": 0.575})["passed"] is False, "-2.6 pt"
    assert _gate("mmlu_holds", {**mm, "mmlu_after": 0.582})["passed"] is True, "-1.9 pt"
    capsys.readouterr()


def test_ce_falls_is_not_measured_on_the_rl_path(capsys):
    """`ce_first` is written by the SFT loop only, so on an RL run the gate has no
    threshold -- and a missing threshold is a vacuous pass, which recorded `passed` over
    nothing. `skipped` says what is true. The gate stays live where both values exist,
    so the rising arm must still fail: a skip that also swallowed a real regression would
    be the same defect in the other direction.
    """
    rl = _gate("ce_falls", {"ce_last": 1.22, "tied_group_fraction": 0.3})
    assert rl["skipped"] is True and rl["passed"] is None, rl
    capsys.readouterr()

    assert _gate("ce_falls", {"ce_last": 1.22, "ce_first": 1.90})["passed"] is True
    capsys.readouterr()
    rising = _gate("ce_falls", {"ce_last": 1.90, "ce_first": 1.22})
    assert rising["skipped"] is False and rising["passed"] is False, rising
    capsys.readouterr()


def test_mcnemar_is_paired_and_reports_no_discordance_distinctly():
    """The paired test over per-question rows, and the two ways it can decline.

    `b + c == 0` (both arms agree on every question) is NOT the same as unpairable
    input: the first is a real result with nothing to resolve, the second is no result.
    Conflating them would let a broken pairing read as perfect agreement.
    """
    from tilerl.cli import _mcnemar

    def rows(flags):
        return [{"dataset": "gsm8k", "i": i, "correct": c} for i, c in enumerate(flags)]

    # 3 wrong->right, 1 right->wrong, over 8: delta = (3-1)/8, se = sqrt(4)/8
    before = rows([True, False, False, False, True, True, True, True])
    after = rows([False, True, True, True, True, True, True, True])
    r = _mcnemar(before, after)
    assert (r["n"], r["b"], r["c"]) == (8, 1, 3), r
    assert r["delta"] == 0.25 and abs(r["se"] - 0.25) < 1e-12, r
    assert abs(r["z"] - 1.0) < 1e-12, r

    # Identical arms: a result, not a failure -- se and z are None, delta is exactly 0.
    same = _mcnemar(before, list(before))
    assert same["b"] == same["c"] == 0 and same["delta"] == 0.0
    assert same["se"] is None and same["z"] is None, same

    # Unpairable: different question sets. Must be None, not a zero-discordance dict.
    assert _mcnemar(before, rows([True] * 7)) is None
    assert _mcnemar([], list(before)) is None
    # A row without `i` cannot be paired, so it is dropped rather than misaligned.
    assert _mcnemar([{"dataset": "gsm8k", "correct": True}], list(before)) is None


def test_validity_gates_cannot_make_p1_read_pass(capsys):
    """A validity gate is recorded apart from the verdict, and never contributes to it.

    `reward_rises` is the case: reward is what GRPO optimizes, so it rising is the
    optimizer working, and it is compatible with a falling eval. Both directions are
    asserted -- a validity failure must not hide a verdict pass, and a verdict failure
    must still fail regardless of validity.
    """
    from tilerl.ledger import verdict_of

    base = {"gsm8k_before": 181, "gsm8k_before_total": 500, "gsm8k_after_total": 500,
            "mmlu_before": 0.601, "mmlu_after": 0.60}

    def run(metrics):
        from tilerl.cli import _finish

        m = new_manifest("train", {"steps": 100, "source": "tiny"}, [])
        m["metrics"] = dict(metrics)
        with contextlib.suppress(SystemExit):
            _finish(m, as_json=True)
        capsys.readouterr()
        return m

    # Verdict passes (+25 of 500, MMLU held), validity fails (reward fell, groups tied).
    m = run({**base, "gsm8k_after": 206, "reward_first": 0.4, "reward_last": 0.3,
             "tied_group_fraction": 0.9})
    assert verdict_of(m, "verdict") is True, m["gates"]
    assert verdict_of(m, "validity") is False, m["gates"]
    assert not gates_pass(m), "an uninterpretable run still exits non-zero"
    assert format_run(m).split()[3] == "novalid", format_run(m)

    # Verdict fails (+1 of 500), validity passes. The verdict must not be rescued.
    m = run({**base, "gsm8k_after": 182, "reward_first": 0.3, "reward_last": 0.4,
             "tied_group_fraction": 0.1})
    assert verdict_of(m, "verdict") is False, m["gates"]
    assert verdict_of(m, "validity") is True, m["gates"]
    assert format_run(m).split()[3] == "FAIL", format_run(m)

    # None, the third state: verdict gates all SKIPPED is "P1 untested", not "P1 failed".
    # Mutation-driven -- dropping the `if scored else None` guard survived every case
    # above, because `all([])` is True and no assertion distinguished it from a pass.
    # `steps == 0` is the real path there: it skips every gate, which is what a
    # smoke-test invocation does, and a `None` collapsed to True would report that run as
    # having passed P1. Note an ABSENT metric is not a skip -- it vacuous-passes by
    # design (`v is None or t is None`), so the skip has to come from the run's shape.
    from tilerl.cli import _finish

    m0 = new_manifest("train", {"steps": 0, "source": "tiny"}, [])
    m0["metrics"] = dict(base, gsm8k_after=206)
    with contextlib.suppress(SystemExit):
        _finish(m0, as_json=True)
    capsys.readouterr()
    assert all(g["skipped"] for g in m0["gates"]), m0["gates"]
    assert verdict_of(m0, "verdict") is None, m0["gates"]
    assert verdict_of(m0, "validity") is None, m0["gates"]

    # Every gate carries a class, so no consumer has to default one.
    assert all("kind" in g for g in m["gates"]), m["gates"]
    kinds = {g["name"]: g["kind"] for g in m["gates"]}
    assert kinds["gsm8k_improves"] == kinds["mmlu_holds"] == "verdict", kinds
    assert kinds["reward_rises"] == kinds["groups_untied"] == "validity", kinds
