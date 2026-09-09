#!/usr/bin/env bash
# One-time dev setup: install deps and enable the versioned git hooks.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --dev
git config core.hooksPath .githooks
echo "setup: deps synced, core.hooksPath=.githooks (pre-push runs ruff)"
