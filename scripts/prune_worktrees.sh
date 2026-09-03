#!/usr/bin/env bash
# Remove worktrees sitting exactly on a merged PR's head, with nothing to lose.
# Dry-run unless --apply. Sessions share this checkout, so an uncommitted file
# may be a peer's work in progress: dirty is a keep, never a --force.
set -uo pipefail

here=$(pwd -P)
junk='(^|/)(\.venv|__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|node_modules|\.DS_Store|\.coverage)/?$'
git fetch -q origin main
# the oid, not the name: a branch name is reused, gets commits after its PR
# merges, or belongs to a fork, and all three still match by name. The oid list
# covers rebase- and squash-merges, where the head never becomes main's ancestor.
merged=$(gh pr list --state merged --limit 200 --json headRefOid -q '.[].headRefOid' | sort -u)

git worktree list --porcelain | awk '
  /^worktree /{d=substr($0,10); b=""; l=""}
  /^branch /{b=substr($0,8)}
  /^locked/{l="locked"}
  /^$/{if(d!="") print d"\t"b"\t"l; d=""}
  END{if(d!="") print d"\t"b"\t"l}
' | while IFS=$'\t' read -r dir ref lock; do
  # a linked worktree's .git is a file; only the main checkout's is a directory
  [ -d "$dir/.git" ] && { echo "skip     $dir  (main checkout)"; continue; }
  case "$here" in "$dir"*) echo "skip     $dir  (you are in it)"; continue ;; esac
  [ -n "$lock" ] && { echo "skip     $dir  (locked)"; continue; }
  b=${ref#refs/heads/}
  [ -n "$b" ] || { echo "skip     $dir  (detached)"; continue; }
  head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
  git -C "$dir" merge-base --is-ancestor "$head" origin/main 2>/dev/null ||
    grep -qxF -- "$head" <<<"$merged" ||
    { echo "keep     $dir  ($b: holds commits not in main)"; continue; }
  n=$(git -C "$dir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" = 0 ] || { echo "keep     $dir  ($b: $n uncommitted)"; continue; }
  # status never reports ignored paths and worktree remove deletes them anyway, so
  # a worktree holding runs/ or .claude/ reads as 0 changes. Allowlist, not
  # blocklist: an ignored path nobody listed here is a keep.
  ig=$(git -C "$dir" status --porcelain --ignored 2>/dev/null | sed -n 's|^!! ||p' | grep -Ev "$junk")
  [ -z "$ig" ] || { echo "keep     $dir  ($b: ignored $(echo $ig | head -c 60))"; continue; }
  if [ "${1:-}" = --apply ]; then
    git worktree remove "$dir" && echo "removed  $dir  ($b)"
  else
    echo "REMOVE   $dir  ($b)"
  fi
done
