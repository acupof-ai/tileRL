# `uv run` builds a venv the pod driver cannot run

Date: 2026-09-09
Status: closed (workaround: launch with the tl013 interpreter; AGENTS.md unchanged pending ckl)

## Context

A training launch on the GPU pod died at `torch.cuda.init` with
`The NVIDIA driver on your system is too old (found version 12090)`.
The first hypothesis was a deleted venv — a `.venv` appeared to have been
"rebuilt at 06:04". That hypothesis was wrong: **no one deleted anything.
The venv had never existed before that moment.**

## Root cause

`uv run` ignores `PATH` and builds its own `.venv` from `uv.lock`:
python 3.11 with torch **2.13.0+cu130** from pypi. The pod driver is
535.161.08; CUDA 13.0 needs a newer driver, so torch dies the first time
it touches the GPU.

The environment that actually works on the pod is `/work/tl013`
(torch **2.11.0+cu129**, python 3.12.3). `pod_run.sh` puts
`/work/tl013/bin` first on `PATH` and points `PYTHONPATH` at the tree's
`src` — but `uv run` bypasses `PATH` entirely, so the working
environment is invisible to it. The live p1gate run uses tl013 plus
`PYTHONPATH`, which is why it was unaffected.

The core fact, not a footnote: **`uv.lock` has pinned torch 2.13.0 since
the uv scaffolding commit (1ce3e50)** — `uv sync` has *never* installed
a runnable torch on this pod. Pod reproducibility has rested on a
hand-built environment (`/work/tl013`, created by uv 0.11.21 on Sep 2,
`include-system-site-packages = true`) that no document records.

The failure is also late: packaging, venv creation, and model load all
succeed. Only `torch.cuda.init` dies — an hour after the launch started.

## Fix

On the pod, launch with the tl013 interpreter directly:

```
/work/tl013/bin/python -m tilerl.cli train ...
```

or go through `pod_run.sh`, which prepends tl013 and sets
`PYTHONPATH` itself. Do not use `uv run` on the pod.

## Why no gate catches this

CI is CPU-only (`TILERL_TARGET=cpu`). The breakage exists only on a
machine with a GPU, and only at `torch.cuda.init` — no automated check
in the repo can see it.

## Rule

A "standard command" is standard only after it has been verified on the
machine it actually runs on. The repo's documented launch
(AGENTS.md, Build & run: `uv run tilerl train --recipe X`) is broken on
the only GPU machine, and CI's green says nothing about it — a gate that
cannot exercise a path cannot certify it. AGENTS.md is deliberately left
unchanged: it is the shared contract every session reads, and adding the
pod exception is ckl's call, not a drive-by edit.
