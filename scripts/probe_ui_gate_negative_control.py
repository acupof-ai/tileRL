"""Negative control: run the JS reference gate against HEAD~1, where addThinking was undefined.

The unit-level negative control in tests/test_chat_ui.py proves the regex can match a
synthetic string. This proves the gate catches the ACTUAL bug that shipped.
"""
import re
import subprocess
import sys

sys.path.insert(0, "tests")
from test_chat_ui import _script, _unresolved  # noqa: E402

REV = "HEAD~1"


def main() -> None:
    # _CHAT_UI moved to ui_assets.py; the buggy revision predates that, so try both
    # rather than pin one -- this reads a commit from before the split by design.
    for path in ("src/tilerl/ui_assets.py", "src/tilerl/server.py"):
        r = subprocess.run(["git", "show", f"{REV}:{path}"], capture_output=True, text=True)
        ui = re.search(r'_CHAT_UI = """(.*?)"""', r.stdout, re.S) if r.returncode == 0 else None
        if ui:
            break
    assert ui, f"could not find _CHAT_UI in {REV} (looked in ui_assets.py and server.py)"
    unresolved = _unresolved(_script(ui.group(1)))
    print(f"{REV} unresolved: {sorted(unresolved)}")
    assert "addThinking" in unresolved, (
        "the gate does NOT catch the bug it was written for -- it is decoration"
    )
    print("negative control PASSED: the gate fails on the version that shipped broken")


if __name__ == "__main__":
    main()
