"""Negative control: run the JS reference gate against HEAD~1, where addThinking was undefined.

The unit-level negative control in tests/test_chat_ui.py proves the regex can match a
synthetic string. This proves the gate catches the ACTUAL bug that shipped.
"""
import re
import subprocess
import sys

sys.path.insert(0, "tests")
from test_chat_ui import _script, _unresolved  # noqa: E402


def main() -> None:
    old = subprocess.run(
        ["git", "show", "HEAD~1:src/tilerl/server.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    ui = re.search(r'_CHAT_UI = """(.*?)"""', old, re.S)
    assert ui, "could not find _CHAT_UI in HEAD~1"
    unresolved = _unresolved(_script(ui.group(1)))
    print(f"HEAD~1 unresolved: {sorted(unresolved)}")
    assert "addThinking" in unresolved, (
        "the gate does NOT catch the bug it was written for -- it is decoration"
    )
    print("negative control PASSED: the gate fails on the version that shipped broken")


if __name__ == "__main__":
    main()
