# A failed stamp wrote a blank commit, not "unknown" — 2026-09-06

> Status: Shipped in the same change. Both halves fixed, each with a red control.

## Context

The pod is not a git repo, so a bench row's provenance comes from a stamp:
`pod_sync.sh` writes `git rev-parse --short HEAD` into `.synced_commit` before the
tarball goes over, and `bench_harness._git_commit` reads it when `git` fails on the
far side. Every row in `bench-baseline.json` carries the `commit` that comes out of
that pair.

I was reading the script for an unrelated reason — I had just changed its wipe — and
the stamp line's `> file ... || true` looked wrong for a reason the `|| true` hides.

## Root Cause

**Two halves of one defect, on opposite sides of the seam.**

The write: `git -C "$ROOT" rev-parse --short HEAD > "$ROOT/.synced_commit" 2>/dev/null || true`.
The shell creates and **truncates** the redirect target before `git` runs, so a git
failure leaves the file **empty** rather than leaving the previous stamp. `|| true`
then swallows the exit code, so the sync continues.

The read: `return stamp.read_text().strip() if stamp.exists() else "unknown"`.
The fallback keys on `exists()`, and an empty file exists. So an empty stamp reads
back as `''`, and `Gate.check` writes `{"tok_s": ..., "commit": "", "date": ...}` —
a row that looks provenanced and is not. The `"unknown"` branch, which exists for
exactly this case, is unreachable whenever the file is present-but-empty.

Probe, each half against the shipped code with the old spelling as a control:

| half | condition | old | fixed |
|---|---|---|---|
| read | stamp empty | `''` | `unknown` |
| read | stamp whitespace | `''` | `unknown` |
| read | stamp absent | `unknown` | `unknown` |
| read | stamp `faae3c8` | `faae3c8` | `faae3c8` |
| write | git fails, previous stamp `faae3c8` | **blanked** | `faae3c8` kept |
| write | git succeeds | new sha | new sha |

## It never fired, and that is the honest scope

**0 of 41 rows in `bench-baseline.json` carry a blank commit.** Five carry the string
`unknown` (`decode-kv/{d512,d2048,d8192,d32768}-b1/sm90` and `d8192-b8/sm90`), which
is the intended value for a row measured before the stamp existed, not this defect's
signature.

Nor do I have a run where it could have fired: the stamp is written on the Mac, inside
the checkout, where `rev-parse` does not fail. The reachable path is someone running
`pod_sync.sh` from a copy of the tree that is not a git checkout — plausible, since the
script's own tarball is exactly such a copy, but not something I can point at having
happened. **This is a latent defect found by reading, and the entry claims no more than
that.**

What makes it worth the fix rather than a note: the failure is silent on both sides,
and the artifact it corrupts is the one that carries provenance for every perf claim.
A blank `commit` in a baseline row is indistinguishable from a row nobody stamped, and
the next reader has no way to tell which happened.

## Fix

`pod_sync.sh` writes only when git succeeded:

```bash
if sha=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null); then
  printf '%s\n' "$sha" > "$ROOT/.synced_commit"
fi
```

`bench_harness._git_commit` keys on content, not existence:

```python
return (stamp.read_text().strip() if stamp.exists() else "") or "unknown"
```

`tests/test_bench_commit_stamp.py` gates both. Each half's assertion has a **red
control that runs the old spelling** — the `exists()` return line and the one-line
redirect — so neither green can be vacuous. The shell half extracts the block from
`pod_sync.sh` rather than copying it, so a rewrite there is what the test sees.

One control was itself broken and the fix's own comment is why: my "did the control
apply" guard was `'or "unknown"' not in src`, which matched the **comment** I had just
added above the return line, so the control reported not-applied while being applied
correctly. Replaced with `src.count(<the exact return line>) == 1`, which also fails
loudly if the line moves.

## Rule

**A redirect truncates before the command runs, so `cmd > f || true` turns a failure
into an empty file, not an unchanged one.** Any fallback that keys on `exists()` is
then unreachable in exactly the case it was written for. When a write and its
fallback-on-missing live in different files, check what the failed write actually
leaves behind, not what its exit code says.

Second, from the broken control: **a guard that greps the source can match a comment.**
Assert on the code line itself, and assert its count, so the guard fails when the line
it names is gone.

## Results

No perf change. Provenance only; no measurement to report.
