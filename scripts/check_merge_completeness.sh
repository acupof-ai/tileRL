#!/bin/bash
# Merge completeness check: the merge result must contain every change main
# made since this branch's base. A line main added that is absent from the
# merge result was silently dropped by the three-way merge — the branch never
# contained it, so it cannot be an intentional deletion.
#
# Incidents that motivated this:
#   #364 ate #357's --deterministic (merged, not caught)
#   #342 would eat #367's pytest-xdist (caught by manual review)
#   #384 would eat #383's fix again (caught by manual review)
#
# Fix is always: git rebase origin/main
set -euo pipefail

base=$(git merge-base HEAD origin/main 2>/dev/null) || {
  echo "SKIP: cannot compute merge-base (shallow clone?)"
  exit 0
}

# Files the PR changes (three-dot: changes on HEAD since divergence)
changed_files=$(git diff --name-only "$base...HEAD")

if [ -z "$changed_files" ]; then
  echo "No files changed in PR."
  exit 0
fi

# Compute the merge tree. On conflict git exits non-zero; GitHub blocks those
# merges itself, so we skip.
merge_tree=$(git merge-tree --write-tree origin/main HEAD 2>/dev/null) || {
  echo "SKIP: merge-tree reports conflicts; GitHub will block."
  exit 0
}

missing=0
while IFS= read -r f; do
  # Lines main added to this file since the branch diverged
  main_added=$(git diff "$base..origin/main" -- "$f" \
    | grep '^+' | grep -v '^+++' | sed 's/^+//' || true)
  if [ -z "$main_added" ]; then
    continue
  fi

  merged_content=$(git show "$merge_tree:$f" 2>/dev/null || echo "")

  while IFS= read -r line; do
    if ! printf '%s' "$merged_content" | grep -qF -- "$line"; then
      echo "MISSING in $f: $line"
      commit=$(git log --oneline -1 "$base..origin/main" -- "$f")
      echo "  introduced by: $commit"
      missing=$((missing + 1))
    fi
  done <<< "$main_added"
done <<< "$changed_files"

if [ "$missing" -gt 0 ]; then
  echo ""
  echo "FAIL: $missing line(s) from main are missing in the merge result."
  echo "This branch is based on an older revision of main. Rebase: git rebase origin/main"
  exit 1
fi

echo "OK: all main changes since branch base are present in the merge result."
