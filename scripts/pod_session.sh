#!/usr/bin/env bash
# shellcheck shell=bash
# One tree per session on the pod, sourced by pod_sync.sh, pod_run.sh and pod_fan.sh.
# Why: pod_sync.sh wipes the tree it syncs into, so one shared tree let two sessions revert
# each other (errors/2026-09-07-two-sessions-one-pod-tree.md).

# `s-` because /work already holds ad-hoc `tilerl-<word>` trees a bare prefix would collide with.
POD_TREE_PREFIX="${POD_TREE_PREFIX:-/work/tilerl-s-}"
POD_BASELINE_DIR="${POD_BASELINE_DIR:-/work/tilerl-baseline}"   # outside every tree, so no wipe reaches it

pod_session_name() {  # pod_session_name <repo-root>
  if [ -n "${POD_SESSION:-}" ]; then
    printf '%s\n' "$POD_SESSION"
    return
  fi
  # printf, not `basename | tr -c`: tr translates the trailing newline too and appends a dash.
  printf '%s\n' "$(printf '%s' "$(basename "$1")" | tr -c 'A-Za-z0-9._-' '-')"
}

pod_session_tree() {  # pod_session_tree <repo-root> -- this session's REMOTE_DIR
  printf '%s%s\n' "$POD_TREE_PREFIX" "$(pod_session_name "$1")"
}
