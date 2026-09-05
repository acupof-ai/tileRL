"""Negative control: run the JS reference gate against the commit where addThinking shipped
undefined.

The unit-level negative control in tests/test_chat_ui.py proves the regex can match a
synthetic string. This proves the gate catches the ACTUAL bug that shipped.

The commit is pinned by sha. It was written as `HEAD~1`, which was true on the day it was
written and false on every day after: a relative ref re-aims itself at each new commit, so
this probe's useful life was exactly one commit. Nothing noticed, because it lives in
scripts/ rather than CI -- run against main on 2026-09-05 it failed with "the gate does NOT
catch the bug it was written for", which reads as the gate being broken rather than the
probe pointing somewhere else.

Verified before pinning: 69398d4 is the parent of e954f8c (the fix), and it is the only one
of {69398d4, e954f8c, HEAD~1, origin/main} whose _CHAT_UI leaves addThinking unresolved.
"""
import re
import subprocess
import sys

sys.path.insert(0, "tests")
from test_chat_ui import _script, _unresolved  # noqa: E402

#: `fix(server): render the checkpoint's chat template, and count tokens for real` --
#: the last commit that shipped the chat page calling an undefined addThinking.
BROKEN = "69398d4"


def main() -> None:
    old = subprocess.run(
        ["git", "show", f"{BROKEN}:src/tilerl/server.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    ui = re.search(r'_CHAT_UI = """(.*?)"""', old, re.S)
    # A later refactor may move _CHAT_UI out of server.py; at BROKEN it is there, and this
    # message says which is wrong -- the probe's target, not the repo's present shape.
    assert ui, f"no _CHAT_UI in {BROKEN}:src/tilerl/server.py -- wrong sha, or wrong file"
    unresolved = _unresolved(_script(ui.group(1)))
    print(f"{BROKEN} unresolved: {sorted(unresolved)}")
    assert "addThinking" in unresolved, (
        "the gate does NOT catch the bug it was written for -- it is decoration"
    )
    print("negative control PASSED: the gate fails on the version that shipped broken")


if __name__ == "__main__":
    main()
