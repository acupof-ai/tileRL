"""Card guard: refuse a card not granted to tileRL when a grant ledger exists."""

import json
import subprocess
from pathlib import Path

import pytest

from tilerl.engine import card_guard


def _assignment(tmp_path, cards: dict, note: str = "") -> str:
    p = tmp_path / "card_assignment.json"
    p.write_text(json.dumps({"cards": cards, "note": note}))
    return str(p)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TILERL_CARD_LEND", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_refuses_a_card_granted_to_another_team(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit, match="theirs"):
        card_guard()


def test_refuses_an_unclassified_card(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "LANE 2026-09-09"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit, match="unclassified"):
        card_guard()


def test_allows_our_own_card(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CARD_ASSIGNMENT_JSON",
        _assignment(tmp_path, {"1": "tileRL, by the user directly"}),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    card_guard()


def test_refuses_our_card_when_note_records_a_lend(tmp_path, monkeypatch):
    """cards says ours, but note records a lend — the guard must refuse."""
    monkeypatch.setenv(
        "CARD_ASSIGNMENT_JSON",
        _assignment(
            tmp_path,
            {"1": "tileRL, by the user directly"},
            note="LENT, NOT TRANSFERRED: cards 1 and 3 are lent by tilerl-27 to b0.",
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(SystemExit, match="lent out"):
        card_guard()


def test_allows_our_card_when_note_lends_a_different_card(tmp_path, monkeypatch):
    """Note records a lend, but not for this card — allow."""
    monkeypatch.setenv(
        "CARD_ASSIGNMENT_JSON",
        _assignment(
            tmp_path,
            {"6": "tileRL, by the user directly"},
            note="LENT, NOT TRANSFERRED: cards 1 and 3 are lent by tilerl-27 to b0.",
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    card_guard()


def test_refuses_when_cuda_visible_devices_unset(tmp_path, monkeypatch):
    """On a machine with a grant ledger, unset CUDA_VISIBLE_DEVICES = all cards visible → refuse."""
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    with pytest.raises(SystemExit, match="unset"):
        card_guard()


def test_allows_when_cuda_visible_devices_explicitly_empty(tmp_path, monkeypatch):
    """Explicitly empty = no cards = CPU only → pass."""
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    card_guard()


def test_allows_when_no_grant_ledger_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", str(tmp_path / "nonexistent.json"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    card_guard()


def test_lend_env_var_bypasses_with_a_log_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("TILERL_CARD_LEND", "lends/2026-09-10-p1.md")
    card_guard()
    assert "lend recorded" in capsys.readouterr().err


def _tracked_paths(root: Path, *pattern_args: str) -> set[str]:
    """Tracked .py files under scripts/ and src/ whose HEAD content matches.

    Pattern type is pinned by the caller's flag (-F/-E), never git's default:
    grep.patternType is a global config, and an extended default inverts both
    the gate and the broken-pattern regression test.

    The caller confirms the repo exists first, so rc>1 here is a broken pattern
    or a git failure — never 'no repo' — and must raise: a skip would pass a
    gate that had stopped judging anything (the unbalanced-pattern bug)."""
    proc = subprocess.run(
        ["git", "grep", "-l", *pattern_args, "HEAD", "--", "scripts", "src"],
        cwd=root, capture_output=True, text=True,
    )
    if proc.returncode > 1:
        raise RuntimeError(f"git grep failed rc={proc.returncode}: {proc.stderr.strip()}")
    return {p.removeprefix("HEAD:") for p in proc.stdout.splitlines() if p.endswith(".py")}


def test_every_materialize_call_site_has_a_guard():
    """Every file that calls backend.materialize() must also contain card_guard()
    or build_engine() (which calls card_guard internally).

    This is a file-level string check, NOT an order check: it asserts the guard
    is present in the file, not that it runs before materialize. The pre-fix
    probe_kv_ceiling.py would pass this gate (it had build_engine in the file
    but called materialize first). Order is human-readable at the current ~15
    call sites; an AST-level order check is over-engineering until that grows.

    Coverage: scripts/ and src/ — a new materialize call site in either tree
    must have the guard. Fails when someone adds one without it.

    Both the site list and the guard check read the HEAD commit (git grep),
    not the working tree: the gate judges the committed code a reviewer sees,
    so a tracked file deleted or edited locally neither breaks it nor slips
    past it, and an untracked probe cannot move the verdict."""
    root = Path(__file__).resolve().parent.parent
    try:
        in_repo = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=root, capture_output=True
        ).returncode == 0
    except OSError:
        in_repo = False
    if not in_repo:
        pytest.skip("no .git — tracked-file enumeration unavailable")
    sites = _tracked_paths(root, "-F", ".materialize(")
    guarded = _tracked_paths(root, "-E", "card_guard|build_engine")
    unguarded = sorted(sites - guarded)
    # Lower bound against silent empty enumeration (a broken git query above
    # would make `unguarded` vacuously empty). 15 tracked sites on 2026-09-10;
    # this is not a coverage requirement. The sha distinguishes a broken query
    # from a tree behind main (an old tree legitimately has fewer sites).
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root,
        capture_output=True, text=True,
    ).stdout.strip()
    assert len(sites) >= 12, (
        f"only {len(sites)} tracked materialize sites found at {head} — "
        f"broken enumeration, or a tree behind main?"
    )
    assert not unguarded, (
        f"files calling backend.materialize() without card_guard or build_engine: "
        f"{unguarded}. Add card_guard() before the materialize call."
    )


def test_tracked_paths_raises_on_a_broken_pattern():
    """A fatal git pattern must raise, not skip: an unbalanced group once made
    git grep return rc=128, which the skip branch ate — the gate went green
    while judging nothing. The repo check and the grep rc are asked separately."""
    root = Path(__file__).resolve().parent.parent
    with pytest.raises(RuntimeError, match="git grep failed"):
        _tracked_paths(root, "-E", ".materialize(")
