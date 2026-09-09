"""Card guard: refuse a card not granted to tileRL when a grant ledger exists."""

import json

import pytest

from tilerl.engine import _card_guard


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
        _card_guard()


def test_refuses_an_unclassified_card(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "LANE 2026-09-09"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(SystemExit, match="unclassified"):
        _card_guard()


def test_allows_our_own_card(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CARD_ASSIGNMENT_JSON",
        _assignment(tmp_path, {"1": "tileRL, by the user directly"}),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _card_guard()


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
        _card_guard()


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
    _card_guard()


def test_refuses_when_cuda_visible_devices_unset(tmp_path, monkeypatch):
    """On a machine with a grant ledger, unset CUDA_VISIBLE_DEVICES = all cards visible → refuse."""
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    with pytest.raises(SystemExit, match="unset"):
        _card_guard()


def test_allows_when_cuda_visible_devices_explicitly_empty(tmp_path, monkeypatch):
    """Explicitly empty = no cards = CPU only → pass."""
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _card_guard()


def test_allows_when_no_grant_ledger_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", str(tmp_path / "nonexistent.json"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    _card_guard()


def test_lend_env_var_bypasses_with_a_log_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CARD_ASSIGNMENT_JSON", _assignment(tmp_path, {"2": "granted to b0"}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("TILERL_CARD_LEND", "lends/2026-09-10-p1.md")
    _card_guard()
    assert "lend recorded" in capsys.readouterr().err
