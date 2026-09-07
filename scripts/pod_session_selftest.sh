#!/usr/bin/env bash
# Two sessions must get two trees, or one session's sync reverts the other's staged branch
# (errors/2026-09-07-two-sessions-one-pod-tree.md).
#
#   scripts/pod_session_selftest.sh     # exits 0, prints PASS
#
# Local, no pod: what is under test is the path composition, not the container plumbing.
set -euo pipefail

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/scripts/pod_session.sh"

fail() { echo "FAIL: $1" >&2; exit 1; }

# ---- 1: the name is per-session, sanitised, and overridable ---------------------------
got=$(pod_session_name "/Users/x/.claude/worktrees/v100-sm70-fp4")
[ "$got" = "v100-sm70-fp4" ] || fail "worktree basename: got [$got]"
got=$(pod_session_name "/x/y/tileRL")
[ "$got" = "tileRL" ] || fail "plain checkout: got [$got]"
# A trailing dash is the shipped bug this catches: `tr -c` translates the newline too.
case "$got" in *-) fail "the name gained a trailing dash: [$got]";; esac
got=$(pod_session_name '/a/b/we ird/na$me')
[ "$got" = "na-me" ] || fail "sanitiser: got [$got]"
got=$(POD_SESSION=explicit pod_session_name /x/y/z)
[ "$got" = "explicit" ] || fail "POD_SESSION override: got [$got]"

# ---- 2: two sessions land in two trees, and neither wipe reaches the other ------------
# pod_sync.sh's wipe verbatim: confined by the cd, so what matters is that the dirs differ.
wipe="find . -mindepth 1 \! -path './runs' \! -path './runs/*' -delete"
for s in alpha beta; do
  dir="$TMP/work/tilerl-$s"
  mkdir -p "$dir/runs"
  printf 'sha-%s\n' "$s" > "$dir/.synced_commit"
  printf '%s\n' "$s" > "$dir/marker"
  printf 'kept\n' > "$dir/runs/manifest.json"
done
# beta syncs: its wipe runs in its own tree only.
( cd "$TMP/work/tilerl-beta" && eval "$wipe" )
[ -f "$TMP/work/tilerl-alpha/marker" ] || fail "beta's wipe deleted alpha's tree"
[ "$(cat "$TMP/work/tilerl-alpha/.synced_commit")" = "sha-alpha" ] \
  || fail "alpha's sha changed under beta's sync"
[ ! -f "$TMP/work/tilerl-beta/marker" ] || fail "beta's own wipe did not run"
[ -f "$TMP/work/tilerl-beta/runs/manifest.json" ] || fail "the wipe took runs/ with it"

# ---- 3: the runner reports the tree it ran in, and its sha ---------------------------
# Emitted from the shipped pod_run.sh, so a number in a log is attributable.
for s in alpha beta; do
  runner=$(REMOTE_DIR="$TMP/work/tilerl-$s" POD_RUN_EMIT_RUNNER=1 \
             bash "$ROOT/scripts/pod_run.sh" t 6 -- true)
  case "$runner" in
    *"pod_run: tree $TMP/work/tilerl-$s sha"*) ;;
    *) fail "$s: the runner does not name its tree: $(printf '%s' "$runner" | head -4)";;
  esac
done
# It must precede anything that can fail. Positioned against the first EXECUTABLE line: a
# line number would break on a comment edit and says nothing about ordering.
runner=$(REMOTE_DIR="$TMP/work/tilerl-alpha" POD_RUN_EMIT_RUNNER=1 \
           bash "$ROOT/scripts/pod_run.sh" t 6 -- true 2>/dev/null)
code=$(printf '%s\n' "$runner" | grep -vE '^\s*(#|$)')
tree_at=$(printf '%s\n' "$code" | grep -n 'pod_run: tree' | head -1 | cut -d: -f1)
[ -n "$tree_at" ] || fail "the runner never prints its tree"
[ "$tree_at" -le 3 ] || fail "the tree line is executable line $tree_at, too late to be read"
# Nothing that can exit non-zero may precede it. `cd` can, and does when the tree is gone,
# which is precisely the case where the sha matters most -- so it may only be preceded by
# set/cd, nothing else.
before=$(printf '%s\n' "$code" | sed -n "1,$((tree_at - 1))p")
printf '%s\n' "$before" | grep -qvE '^(set |cd )' \
  && fail "something other than set/cd precedes the tree line: $before"

# Run it: `unknown` must be visibly different from a real sha, or the line is decoration.
echo_line=$(printf '%s\n' "$code" | grep 'pod_run: tree' | head -1)
printf 'sha-alpha\n' > "$TMP/work/tilerl-alpha/.synced_commit"
out=$( cd "$TMP/work/tilerl-alpha" && bash -c "$echo_line" )
[ "$out" = "pod_run: tree $TMP/work/tilerl-alpha sha sha-alpha" ] || fail "sha not read: [$out]"
rm -f "$TMP/work/tilerl-alpha/.synced_commit"
out=$( cd "$TMP/work/tilerl-alpha" && bash -c "$echo_line" )
case "$out" in *"sha unknown") ;; *) fail "an unsynced tree did not read unknown: [$out]";; esac

# ---- 4: the shared baseline is outside every session tree ---------------------------
# Wiped with the tree otherwise, which is the row that goes missing.
base=$(POD_BASELINE_DIR="$TMP/work/tilerl-baseline" python3 -c "
import sys; sys.path.insert(0, '$ROOT/scripts')
import baseline; print(baseline.REMOTE)")
case "$base" in
  "$TMP/work/tilerl-baseline/bench-baseline.json") ;;
  *) fail "the baseline is not in the shared dir: [$base]";;
esac
for s in alpha beta; do
  case "$base" in *"tilerl-$s"*) fail "the baseline path is inside session $s's tree";; esac
done

# ---- 5: the DEFAULT tree is per-session in all three scripts -------------------------
# The arms above pass REMOTE_DIR explicitly, so a revert to the shared tree left them green:
# the property is the DEFAULT, so assert it with REMOTE_DIR unset.
default_tree() {  # default_tree <script> <session> -- the REMOTE_DIR it would pick
  POD_SESSION="$2" ROOT="$ROOT" bash -c '
    . "$ROOT/scripts/pod_session.sh"
    SESSION="$(pod_session_name "$ROOT")"   # what pod_sync/fan set before their assignment
    # Read the assignment out of the source: running these would talk to the pod.
    eval "$(grep -m1 "^REMOTE_DIR=" "$1")"
    printf "%s\n" "$REMOTE_DIR"
  ' _ "$1" 2>/dev/null
}
for script in pod_run.sh pod_sync.sh pod_fan.sh; do
  a=$(default_tree "$ROOT/scripts/$script" alpha)
  b=$(default_tree "$ROOT/scripts/$script" beta)
  [ "$a" != "$b" ] || fail "$script: two sessions share the default tree [$a] -- a sync in one wipes the other"
  case "$a" in *alpha*) ;; *) fail "$script: the default tree does not name the session: [$a]";; esac
  [ "$a" != "/work/tilerl" ] || fail "$script: the default is still the shared /work/tilerl"
  # A bare `tilerl-<session>` collides with the ad-hoc `tilerl-<word>` trees already on /work.
  case "$a" in "/work/tilerl-s-"*) ;; *) fail "$script: the tree can collide with an ad-hoc /work/tilerl-<name>: [$a]";; esac
done

# ---- 6: a job's own path defaults follow the session tree -----------------------------
# The three bench scripts take `--repo` as the server's cwd and defaulted to the pre-split
# /work/tilerl, so after the split they would have run a peer's tree. They read REMOTE_DIR,
# which every launcher must therefore EXPORT and not merely cd into.
for launcher in pod_run.sh pod_sync.sh pod_fan.sh; do
  grep -qE "(export [^\"']*|[[:space:]])REMOTE_DIR=\\\$?REMOTE_DIR" "$ROOT/scripts/$launcher" \
    || fail "$launcher does not export REMOTE_DIR; a job resolving its own paths falls back to /work/tilerl"
done
for b in bench_ssd_restart bench_write_through bench_tier_wall_clock; do
  # argparse's own default, not a regex over the source: the first attempt matched to the
  # first comma and tried to eval `os.environ.get("REMOTE_DIR"`.
  parser="
import argparse, pathlib, re
src = pathlib.Path('$ROOT/scripts/$b.py').read_text()
m = re.search(r'--repo\", (default=.+?)\)\n', src, re.S)
assert m, 'no --repo argument'
import os
ap = argparse.ArgumentParser()
exec('ap.add_argument(\"--repo\", ' + m.group(1) + ')')
"
  got=$(REMOTE_DIR="$TMP/work/tilerl-s-alpha" python3 -c "$parser
print(ap.parse_args([]).repo)")
  [ "$got" = "$TMP/work/tilerl-s-alpha" ] || fail "$b's --repo default is [$got], not the session tree"
  # With REMOTE_DIR unset the parser must REFUSE. A fallback here is the same trap for a
  # hand-run job, and it produces a number rather than an error.
  if out=$(env -u REMOTE_DIR python3 -c "$parser
print('RESOLVED', ap.parse_args([]).repo)" 2>&1); then
    fail "$b accepted no --repo with REMOTE_DIR unset: $out"
  fi
  case "$out" in *"required"*|*"the following arguments"*) ;;
    *) fail "$b failed for some reason other than a required --repo: $out";; esac
done

echo "PASS: two sessions get two trees, neither wipe reaches the other, each runner names its tree and sha, the baseline sits outside both, all three scripts default per-session, and a job's own paths follow REMOTE_DIR"
