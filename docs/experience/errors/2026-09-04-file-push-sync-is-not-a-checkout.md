# A file-push sync makes the remote tree a mix of commits — sm70, 2026-09-04

## Context

The V100 pod's `tilerl-v100` is **not a git checkout** (`git rev-parse` says "not a git
repo"). Every change reaches it as `ssh v100 'cat > path' < path`, one file at a time,
chosen by hand. That worked while changes were single-file.

A run then died with `AttributeError: 'ModelConfig' object has no attribute 'head_key'`
from `spec.py:300`. `head_key` is a property upstream added in the merge I landed
tonight. I had pushed `spec.py` and `engine.py` — the files I had just edited — and not
`config.py`, which is where the property lives.

## Root Cause

**A hand-picked file push has no notion of a commit, so the remote tree converges to a
mix of every commit any file was pushed from.** The traceback names the *consumer* of
the missing attribute (`spec.py`), not the stale file (`config.py`), so it reads as a bug
in the code I had just written.

Hashing every `.py` on both sides showed the extent: **9 of 36 files differed** —
`reference.py`, `autograd.py`, `cli.py`, `config.py`, `dflash2.py`, `iso.py`, `model.py`,
`recipes.py`, `train.py`. Six of those I had never touched tonight; they were behind
because the merge brought them forward and nothing pushed them. The tree was not "one
commit plus my edits", it was an arbitrary blend.

It also accumulated junk, though not from the mechanism I first blamed: **568 macOS
AppleDouble `._` files** across the tree, 330 of them dated after 09-01. I wrote "one per
file ever pushed from the Mac" — wrong, `cat > path` cannot create a sibling. Their
contents start `\0\5\x16\a\0\2\0\0Mac OS X`, an AppleDouble header, which is what `tar`
or `scp -r` from macOS writes for a file carrying extended attributes. So they came from
a bulk copy, not from the per-file pushes, and the count is 568 rather than the 36 I
happened to see among `.py` files.

## Fix

Hash-diff, then push exactly the difference, then hash-diff again:

    # on the pod
    for f in $(find src packages -name "*.py" -not -name "._*"); do md5sum $f; done | sort -k2
    # locally, compare, push each differing file, re-compare
    # -> "files: 36 36  differing: 0"

The 9 files went over individually (the harness rejects a loop over `ssh` in one
command, which is why they are separate calls in the log).

**What should replace this**: the pod *can* reach GitHub — `git ls-remote` returns
`485ba2e`. So `git clone` + `git checkout <sha>` is available, and it makes every run
name a sha instead of a blend. Not done tonight because the existing tree holds the
TileLang JIT cache the runs depend on and /data00 is at 91%; filed rather than
half-done.

## Rule

**A remote tree synced by file push is not at any commit, and its version cannot be
quoted.** Before believing a remote failure is in the code you just wrote, hash every
source file on both sides — the traceback names the file that *read* the stale value,
never the stale file. One line of output, "differing: 0", is the only statement that
licenses attributing a remote result to a local sha.

Corollary for reporting: I told a peer earlier tonight that "the merged branch runs on
CUDA — it is what these runs have been running". That was false in a way this rule would
have caught: the pod copy is synced by file push, so it was a checkout of nothing. Said
plainly to that peer as a correction rather than left standing.
