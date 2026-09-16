# On the H20 pod, `uv run` picks a cu130 torch the 535 driver cannot load — use /work/tl013 — 2026-09-16

**Status:** closed (a runbook, not a code defect).

## Context

Running the sm90 fused-prelude gate (audit F6) on an H20 in the `sglang-test`
container, `scripts/pod_run.sh ... -- uv run pytest tests/test_attn_prelude_oracle.py`
exited during collection:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090).
torch 2.13.0+cu130
```

The host driver is **535.161.08 (CUDA 12.9)**; the freshly `uv sync --dev`'d
`/work/tilerl-s-tileRl/.venv` had resolved **torch 2.13.0 whose default manylinux wheel
bundles CUDA 13.0**, which refuses to initialize on the 12.9 driver. The lock's
`torch>=2.11.0` is fine for Mac/CI/newer drivers, but a plain `uv sync` on this pod
picks a wheel newer than its driver.

## Root cause

Two different environments, one name:

- `/work/tilerl-s-tileRl/.venv` — what `uv sync --dev` builds; on 2026-09-16 a mirror
  install put torch 2.13.0+cu130 there, incompatible with driver 535.
- `/work/tl013` — the pod's maintained venv with **torch 2.11.0+cu129**, pytest 9.1.0,
  tilelang and the project importable via `PYTHONPATH`; this is the one that matches the
  host CUDA. `pyproject.toml` already notes "the pod's working env (/work/tl013) has
  2.11.0+cu129".

`uv run` always uses the project `.venv`, so it cannot reach tl013 even though
`pod_run.sh` prepends `/work/tl013/bin` to PATH.

## Rule (H20 / sglang-test container)

Run pod tests with the interpreter **directly**, not `uv run`, so the PATH-prepended
tl013 wins:

```bash
# inside the tree on a claimed card (scripts/pod_run.sh already exports
# PATH=/work/tl013/bin, PYTHONPATH=<tree>/src:<tree>/packages/tilerl-kernels/src,
# TILERL_TARGET=cuda, CUDA_VISIBLE_DEVICES=<card>)
python -m pytest tests/<file>.py -v -s        # uses /work/tl013 torch 2.11.0+cu129
# NOT:  uv run pytest ...                      # uses .venv torch 2.13.0+cu130 -> rc "driver too old"
```

Verify the env before a run:

```bash
/work/tl013/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
# 2.11.0+cu129  True   -> correct; an ...+cu130 line is wrong for this host
```

## Also recorded

- Package mirror for installing into a venv at all: direct pypi.org intermittently
  stalls on the large CUDA wheels from this container; the aliyun mirror
  (`https://mirrors.ivolces.com` was missing `huggingface-hub==1.28.0`, aliyun had it)
  is the working source. But whatever a fresh `uv sync` installs there may still be a
  cu130 torch — prefer the maintained `/work/tl013` for running, regardless of how the
  venv was provisioned.
- The card claim worked end to end through the `--lend-ref` bypass (#491): an
  unclassified H20 with 0 MiB used was claimed, ran, and released cleanly.

## Rule in one line

On the H20 pod, **driver 535 = CUDA 12.9, so run `python -m pytest` against
`/work/tl013` (2.11.0+cu129); never `uv run` (its .venv is cu130 and cannot init
CUDA).**
