#!/usr/bin/env bash
# One 27B engine per python process (several engines in one process OOMs a
# 95 GiB card). Runs the four cells, then compares. Card 4 gate for #564.
set -uo pipefail
cd "$(dirname "$0")/.."
D=/work
P="python3 scripts/probe_warm_control.py --steps ${STEPS:-64} --batch ${BATCH:-8}"
$P --cell coldwave --tag A --out $D/warmctl_A.json || exit 10
$P --cell coldwave --tag B --out $D/warmctl_B.json || exit 11
$P --cell warm1              --out $D/warmctl_warm1.json || exit 12
$P --cell cold1              --out $D/warmctl_cold1.json || exit 13
$P --cell compare
