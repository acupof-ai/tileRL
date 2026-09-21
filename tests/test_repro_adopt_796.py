"""Hermetic gates for scripts/repro_adopt_796.py (V100 #796 adoption harness).

The device run boots a 27B serve and is hand-driven by fixkv on the V100; these
gates cover everything that can fail without a GPU: geometry resolution, the
close-tick parser and the five-key positive control, and the --dry-run assembly
that CI executes. They also make the script reachable to the scripts/ closure
audit (import) so it is not bucketed DEAD.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "scripts" / "repro_adopt_796.py"


def _load():
    import sys as _sys
    spec = importlib.util.spec_from_file_location("repro_adopt_796", SRC)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["repro_adopt_796"] = mod  # dataclass resolves class refs by module
    spec.loader.exec_module(mod)
    return mod


M = _load()

# A real remedial-closure tick (one transfer segment + an mmap) vs #796's
# all-zero close tick.
_MOVED = ("[step-timing] tick 7 total=90ms dec=1 pre=0 model=80ms sample=4ms "
          "path=eager sparse=1 pub_cold_transfer=12ms pub_share_hold=0ms "
          "pub_frame_d2h=0ms pub_bounds_d2h=0ms pub_draft_clone=0ms ssd_mmap=3ms")
_ZERO = _MOVED.replace("pub_cold_transfer=12ms", "pub_cold_transfer=0ms").replace(
    "ssd_mmap=3ms", "ssd_mmap=0ms")


def test_close_tick_parser_reads_all_five_keys_and_mmap():
    t = M.parse_close_tick(_MOVED)
    assert t is not None
    assert t["n"] == 7 and t["dec"] == 1
    assert t["pub_cold_transfer"] == 12 and t["ssd_mmap"] == 3
    for k in M.CLOSE_TRANSFER_KEYS:
        assert k in t, k


def test_close_positive_control_separates_moved_from_all_zero():
    moved, zero = M.parse_close_tick(_MOVED), M.parse_close_tick(_ZERO)
    assert M.close_transfer_present([moved])
    # The #796 signature: every close tick all-zero -> no transfer seen.
    assert not M.close_transfer_present([zero])
    assert not M.close_transfer_present([zero, M.parse_close_tick("not a tick")])


def test_geometry_matches_the_796_32k_head():
    g = M.Geometry(words=16000, tokens_per_word=2.002,
                   suffix_words=64, follower_tokens=48)
    d = g.describe()
    # 16000 words measured ~32028 tokens / 2002 full pages on qwen38; the short
    # suffix is a SHORT advance (~128 tokens), not a long decode.
    assert d["head_tokens_est"] == 32032
    assert d["head_pages_est"] == 2002
    assert d["suffix_tokens_est"] == 128
    assert d["follower_gen_tokens"] == 48


def test_cold_fill_state_reports_four_keys_in_gib():
    page = 2 ** 30
    st = {"kv_cold_bytes": page, "kv_cold_ssd_bytes": 0,
          "kv_cold_shared_bytes": 2 * page, "kv_cold_shared_ssd_bytes": 3 * page,
          "kv_cold_shared_pages": 24}
    fill = M.cold_fill_state(st)
    assert fill == {"priv_gib": 1.0, "priv_ssd_gib": 0.0, "shared_gib": 2.0,
                    "shared_ssd_gib": 3.0, "shared_pages": 24}


def test_dry_run_assembles_without_network():
    r = subprocess.run(
        [sys.executable, str(SRC), "--dry-run", "--words", "16000"],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "dry-run OK" in r.stdout
    assert "sparse_prefix_entries" in r.stdout
    assert "pub_draft_clone" in r.stdout
