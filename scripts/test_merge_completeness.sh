#!/bin/bash
# Self-test for check_merge_completeness.sh.
#
# git merge-tree --write-tree is robust: on text files it either produces a
# correct merge (both sides' changes present) or reports a conflict. It does
# not silently drop one side's additions the way GitHub's squash-merge path
# can (#364). So this test verifies two things:
#
# 1. The check PASSES on a branch whose merge result contains main's additions.
# 2. The missing-line logic FIRES when main's addition is absent from the
#    merge result — proven by injecting a tree that lacks it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECK="$SCRIPT_DIR/check_merge_completeness.sh"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"

git init -q
git config user.email t@t.t
git config user.name t

# Base file
cat > f.txt <<'EOF'
line one
line two
line three
line four
line five
EOF
git add f.txt
git commit -qm base

# Main: adds a line
git checkout -q -b feature-a
sed -i '' '3a\
inserted-by-main
' f.txt 2>/dev/null || sed -i '3a inserted-by-main' f.txt
git add f.txt
git commit -qm "main adds a line"

# Branch from OLD base: modifies a DIFFERENT file (clean merge, main's line present)
git checkout -q -b feature-b HEAD~1
echo "other file" > g.txt
git add g.txt
git commit -qm "branch adds a different file"

git update-ref refs/remotes/origin/main feature-a
git checkout -q feature-b

# --- Test 1: check passes on a clean branch ---
if ! bash "$CHECK"; then
  echo "FAIL: check should pass on a clean branch"
  exit 1
fi
echo "PASS: clean branch accepted"

# --- Test 2: missing-line logic fires ---
# Build a tree that has f.txt WITHOUT main's inserted line, simulating the
# silent-revert merge result. The check must detect it.
git checkout -q feature-a
sed -i '' '/inserted-by-main/d' f.txt 2>/dev/null || sed -i '/inserted-by-main/d' f.txt
git add f.txt
tree_oid=$(git write-tree)
git checkout -q f.txt 2>/dev/null; git checkout -q -- f.txt 2>/dev/null || true

# Run the check's core logic against the injected tree
base=$(git merge-base feature-b origin/main)
main_added=$(git diff "$base..origin/main" -- f.txt \
  | grep '^+' | grep -v '^+++' | sed 's/^+//' || true)
merged_content=$(git show "$tree_oid:f.txt" 2>/dev/null || echo "")

missing=0
while IFS= read -r line; do
  if ! printf '%s' "$merged_content" | grep -qF -- "$line"; then
    missing=$((missing + 1))
  fi
done <<< "$main_added"

if [ "$missing" -eq 0 ]; then
  echo "FAIL: missing-line logic did not fire on a tree lacking main's addition"
  exit 1
fi
echo "PASS: missing-line logic fires ($missing line(s) detected)"

echo "OK: merge completeness check self-test passed"
