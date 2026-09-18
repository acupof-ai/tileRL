# Serving tileRL on an H20 (sm90)

The sparse long-context 27B on an H20 card, inside the `sglang-test` pod. The
supervisor is [`scripts/serve_h20.sh`](../scripts/serve_h20.sh); it is the sm90
analogue of the V100 hybrid supervisor, run **through `scripts/pod_run.sh`**,
never launched by hand.

Status: **written and hermetically gated, but not runnable today.** The
script and its fuse/dry-run gates are on main. H20 was stopped by decision on
2026-09-16 20:40: the `sglang-test` pod object was deleted and its `emptyDir`
`/work` wiped (the `/work/tilerl-s-*` trees, `/work/tl013`, the 27B checkpoint),
so no boot is scheduled until the pod is rebuilt, a tree re-synced, `/work/tl013`
restored and the checkpoint re-uploaded. See
[experience/PENDING-REMOTE-CARDS.md](experience/PENDING-REMOTE-CARDS.md).

## From a fresh pod (one-time bootstrap)

The scripts assume you are already inside the `sglang-test` pod with a card
grant. If the pod was recreated, do this sequence first (read-only recon on
2026-09-17 confirmed which pieces survive).

1. **Pod + card grant — platform/ckl, not this repo.** The sglang-test
   container is scheduled outside the repo; there is no create-pod script. The
   node shows 8× H20, but the card-assignment ledger lives inside the pod at
   `/work/aupai/runs/card_assignment.json`, so confirm the grant / get a fresh
   `--lend-ref` before claiming. Do not `crictl run` a container by hand.

2. **Checkpoint is on host NVMe at `/data00/Qwen3.8-27B-NVFP4`** (22 GiB:
   `model.safetensors` + in-directory `model_mtp.safetensors` + configs).
   Confirm it is mounted at that path inside the new pod. `serve_h20.sh`
   defaults `SERVE_CKPT_DIR` to `/work/tilerl-ckpt/...`; when only `/data00`
   is mounted, point it there explicitly (command below).

3. **Sync a tree, then build `/work/tl013`** (torch 2.11.0+cu129). The
   interpreter does not survive a pod recreate, but it is now scripted:
   [`scripts/h20_tl013_setup.sh`](../scripts/h20_tl013_setup.sh). It is
   idempotent, builds a CPython-3.12 `--system-site-packages` venv that inherits
   the image's cu129 torch, installs tilelang 0.1.13 over the image's 0.1.8, and
   **fail-closed-verifies** torch is `+cu129` (a cu130 build only fails after a
   model load). Never `uv run`.

   ```
   POD_SESSION=h20 scripts/pod_sync.sh                       # tar main -> /work/tilerl-s-h20
   POD_SESSION=h20 scripts/pod_sync.sh 'bash scripts/h20_tl013_setup.sh'
   POD_SESSION=h20 scripts/pod_sync.sh 'bash scripts/serve_h20.sh --dry-run'  # expect exit 0
   ```

   `pod_sync.sh` takes the remote command as its first positional argument (it
   parses only `--session`/`run`); there is NO `--` separator here — a bare `--`
   would run as a literal command on the pod (exit 127) while the sync still
   succeeds, silently skipping setup. (`pod_run.sh` is different and DOES use `--`
   before its command.)

   `POD_SESSION=h20` is what pins the remote tree to `/work/tilerl-s-h20` for
   both `pod_sync.sh` and `pod_run.sh` (the tree name derives from it, not a
   `--session` flag — pod_run has none). On the laptop, sessions use one git
   worktree per branch under `.claude/worktrees/`;
   [`scripts/prune_worktrees.sh`](../scripts/prune_worktrees.sh) dry-runs
   (default) the worktrees sitting exactly on a merged PR head and deletes only
   with `--apply`, keeping a dirty tree (a peer's work in progress) by design.
   PyPI stalls on large wheels from this
   pod; the setup defaults to the `mirrors.aliyun.com` index (the ivolces mirror
   lacks huggingface-hub 1.28.0). Override with `PIP_INDEX_URL` /
   `TL013_TORCH_INDEX` if the network differs.

4. **Launch**, pointing at the `/data00` checkpoint only if `/work/tilerl-ckpt`
   is absent:

   ```
   POD_SESSION=h20 scripts/pod_run.sh --lend-ref "<ckl grant>" h20serve <card> -- \
     SERVE_CKPT_DIR=/data00/Qwen3.8-27B-NVFP4 bash scripts/serve_h20.sh
   ```

   (`env VAR=val bash …` passes the override into the supervisor. Omit the
   `SERVE_CKPT_DIR` prefix when the checkpoint is mounted at the default path.)


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

To watch a `/work` pod log as an event stream from the laptop (a `tn exec`
compound command that a `Monitor` prompt cannot carry), use
[`scripts/pod_tail.sh`](../scripts/pod_tail.sh)
(`scripts/pod_tail.sh <log-basename> [grep-ere] [poll-s] [rounds]`); it prints
only new lines and defaults to a pattern that stays loud through a crash, so
silence means still-running.

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
container restart, but not a pod deletion: the 2026-09-16 20:40 shutdown took
`/work` wholesale, so the cold-tier probes must be re-run, not reused. The V100
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
