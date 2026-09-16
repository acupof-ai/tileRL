# Serving tileRL on an H20 (sm90)

The sparse long-context 27B on an H20 card, inside the `sglang-test` pod. The
supervisor is [`scripts/serve_h20.sh`](../scripts/serve_h20.sh); it is the sm90
analogue of the V100 hybrid supervisor, run **through `scripts/pod_run.sh`**,
never launched by hand.

Status: **written and hermetically gated, awaiting a real serve window.** The
script and its fuse/dry-run gates are on main; an actual boot on a named H20
card is scheduled after the in-flight serve PRs land.

## Run it through pod_run

```
scripts/pod_run.sh [--wait] [--lend-ref "<ledger ref>"] h20serve <card> -- \
    bash scripts/serve_h20.sh
```

`pod_run.sh` supplies everything the V100 script hard-codes: the card claim and
reap, the pod-synced tree (`/work/tilerl-s-<session>`), `PYTHONPATH`, and
`/work/tl013/bin` on `PATH`. Do **not** use `uv run`: a fresh sync installs a
cu130 torch the H20 host's 12.9 driver (535.161.08) cannot load. The maintained
interpreter is `/work/tl013` (torch 2.11.0+cu129), invoked as plain `python3` —
see
[experience/errors/2026-09-16-h20-pod-uses-tl013-cu129-not-uv-run-cu130.md](experience/errors/2026-09-16-h20-pod-uses-tl013-cu129-not-uv-run-cu130.md).

The launcher returns as soon as the job is claimed; tail `/work/h20serve.log`
(the pod_run job log) and `/work/serve_h20.log` (the server log). Pass `--wait`
to block on it. Stop by killing the pod_run job; the supervisor trap releases
the GPU.

## What it serves

`tilerl serve --model qwen38-27b` with the sparse+d1+decode-graph configuration:

- `--sparse-k 128 --sparse-min-tokens 8192` — hybrid: short prompts dense, long
  prompts sparse.
- `--draft <ckpt>/model_mtp.safetensors --depth 1` — one speculative draft token.
- `--decode-graph` — captured decode on sm90 (the auto path is eager on sm70).
- `--slots 8 --max-batch 8 --max-ctx 131072`.
- cold spill tier on by default.

Every value is overridable by env; `bash scripts/serve_h20.sh --dry-run` prints
the resolved serve argv without touching a GPU. The draft lives **inside the
checkpoint directory** (`$TILERL_QWEN38_SOURCE/model_mtp.safetensors`), not in a
separate `mmlu-assets` path as on the V100 host.

## Cold spill path

The default spill file is `/work/sparse_cold_h20.bin` (8 GiB host budget +
8 GiB spill). `/work` is the pod's writable 2 TB scratch that survives a
container restart and already holds the cold-tier probes — the V100
host-specific SSD paths do not apply. The mounted NVMe `/mnt/data02` (3.5 TB) is
root-0755 at the mount point; set `SERVE_COLD_SSD` to a writable subdirectory
there to use it. Set `SERVE_COLD_SSD=""` to run sparse with a host-only cold
budget and no spill file.

## Liveness and the restart fuse

The same out-of-process guard as V100,
[`scripts/serve_liveness.py`](../scripts/serve_liveness.py): two consecutive
failed `/health` polls, a fatal CUDA marker, or three failed short completions
with a free slot request a restart; the supervisor kills and reboots the child.
Five restarts inside ten minutes trip the fuse and the supervisor stays down
(exit 2) — a crash burst that fast is a finding, not something to paper over.
Warmup (`scripts/serve_warmup_hybrid.py`) captures both modes before traffic.
