#!/usr/bin/env bash
# Remove worktrees whose branch has a merged PR and whose tree is clean.
# Dry-run unless --apply. Sessions share this checkout, so an uncommitted file
# may be a peer's work in progress: dirty is a keep, never a --force.
set -uo pipefail

root=$(git rev-parse --show-toplevel) || exit 1
here=$(pwd -P)
merged=$(gh pr list --state merged --limit 200 --json headRefName -q '.[].headRefName' | sort -u)

git worktree list --porcelain | awk '
  /^worktree /{d=substr($0,10); b=""; l=""}
  /^branch /{b=substr($0,8)}
  /^locked/{l="locked"}
  /^$/{if(d!="") print d"\t"b"\t"l; d=""}
  END{if(d!="") print d"\t"b"\t"l}
' | while IFS=$'\t' read -r dir ref lock; do
  [ "$dir" = "$root" ] && continue
  case "$here" in "$dir"*) echo "skip     $dir  (you are in it)"; continue ;; esac
  [ -n "$lock" ] && { echo "skip     $dir  (locked)"; continue; }
  b=${ref#refs/heads/}
  [ -n "$b" ] || { echo "skip     $dir  (detached)"; continue; }
  grep -qxF -- "$b" <<<"$merged" || { echo "keep     $dir  ($b: no merged PR)"; continue; }
  n=$(git -C "$dir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 0 ] || { echo "keep     $dir  ($b: $n uncommitted)"; continue; }
  if [ "${1:-}" = --apply ]; then
    git worktree remove "$dir" && echo "removed  $dir  ($b)"
  else
    echo "REMOVE   $dir  ($b)"
  fi
done
