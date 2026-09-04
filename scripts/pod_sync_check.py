"""Is the pod's source tree the same as this checkout? Answer before trusting a run.

The V100 pod's ``tilerl-v100`` is not a git checkout, and every change reaches it as a
hand-picked ``ssh v100 'cat > path' < path``. A file push has no notion of a commit, so
the remote tree converges to a MIX of every commit any file was pushed from, and its
version cannot be quoted.

Measured 2026-09-04: a run died with ``AttributeError: 'ModelConfig' object has no
attribute 'head_key'`` from ``spec.py:300``. I had pushed spec.py and engine.py -- the
files I had just edited -- while ``head_key`` lives in config.py, which a merge had moved
forward and nothing pushed. **9 of 36 source files differed**, six of them files I had not
touched at all. The traceback named the file that READ the stale value, never the stale
file, so it read as a bug in the code I had just written
(errors/2026-09-04-file-push-sync-is-not-a-checkout.md).

Usage:
    uv run python scripts/pod_sync_check.py            # report
    uv run python scripts/pod_sync_check.py --push     # push what differs, then re-check

The pod CAN reach GitHub (``git ls-remote`` resolves), so a clone plus a checkout would
retire this script. Until then this is the only statement that licenses attributing a
pod result to a local sha.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

POD = "v100"
POD_ROOT = "/data00/home/chenkailun.c/tilerl-v100"
TRACKED = ("src", "packages")
SSH = "/usr/bin/ssh"  # bare `ssh` is a different binary on this machine


def local_hashes(root: Path) -> dict[str, str]:
    out = {}
    for d in TRACKED:
        for f in sorted((root / d).rglob("*.py")):
            if f.name.startswith("._"):
                continue
            rel = str(f.relative_to(root))
            out[rel] = hashlib.md5(f.read_bytes()).hexdigest()
    return out


def pod_hashes() -> dict[str, str]:
    # -not -name '._*' matters: a bulk scp/tar from macOS leaves 568 AppleDouble files
    # in this tree, and they are not source.
    cmd = (f"cd {POD_ROOT} && find {' '.join(TRACKED)} -name '*.py' -not -name '._*' "
           "-exec md5sum {} +")
    r = subprocess.run([SSH, POD, cmd], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"pod hash failed: {r.stderr.strip()[:300]}")
    out = {}
    for line in r.stdout.splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            out[f.strip()] = h
    return out


def push(root: Path, rel: str) -> None:
    with (root / rel).open("rb") as fh:
        r = subprocess.run([SSH, POD, f"cat > {POD_ROOT}/{rel}"], stdin=fh,
                           capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"push {rel} failed: {r.stderr.strip()[:200]}")


def report(loc: dict[str, str], pod: dict[str, str]) -> list[str]:
    differ = sorted(f for f in loc if loc[f] != pod.get(f))
    missing = sorted(f for f in loc if f not in pod)
    extra = sorted(f for f in pod if f not in loc)
    print(f"local {len(loc)} files, pod {len(pod)}, differing {len(differ)}")
    for f in differ:
        print(f"  {'ABSENT' if f in missing else 'differs'}  {f}")
    for f in extra:
        print(f"  only on pod  {f}")
    return differ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true",
                    help="push every differing file, then re-check")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    loc = local_hashes(root)
    differ = report(loc, pod_hashes())
    if not differ:
        print("IN SYNC -- a pod result may be attributed to this checkout")
        return 0
    if not args.push:
        print("\nOUT OF SYNC -- do not attribute a pod result to this checkout. "
              "Re-run with --push.")
        return 1
    for f in differ:
        push(root, f)
        print(f"pushed {f}")
    # Re-check rather than trust the pushes: a truncated write leaves a file that
    # imports and misbehaves, which is the failure this script exists to prevent.
    if report(loc, pod_hashes()):
        print("\nSTILL OUT OF SYNC after pushing")
        return 1
    print("IN SYNC")
    return 0


def _self_check() -> None:
    """`report` must call a missing file out, not silently treat it as matching.

    Runs without the pod. The bug this guards is the one the entry records: an absent
    remote file is the WORST case (the import resolves to something older) and a
    dict-get default of None would make it compare unequal but print as "differs",
    hiding that it is not there at all.
    """
    loc = {"a.py": "1", "b.py": "2", "c.py": "3"}
    pod = {"a.py": "1", "b.py": "9"}  # b differs, c absent
    differ = report(loc, pod)
    assert differ == ["b.py", "c.py"], differ
    assert report(loc, dict(loc)) == [], "an identical tree must report nothing"
    print("pod_sync_check: report OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        raise SystemExit(main())
