# The cu129 index redirect cannot run on the pod

Date: 2026-09-09
Status: closed (partial revert in PR #404)

## Context

#342 pinned `torch==2.13.0` as a direct dependency and redirected linux
to the cu129 index (`download.pytorch.org/whl/cu129`). The motivation
was sound: PyPI's torch 2.13.0 is built for CUDA 13.0, and the pod
driver (535.161.08) supports up to CUDA 12.9, not 13.0. The cu129
index has torch 2.13.0+cu129, which should match the driver.

The pod acceptance failed. Three independent findings, each sufficient
to block the change:

## Root cause

### 1. PyPI CDN is unreachable from the pod

`pypi.org` (the JSON API) returns 200, but actual file downloads from
`files.pythonhosted.org` hang — no bytes, no timeout, no error. The
base URL answering is not the CDN answering. Verified with a range GET
and a positive control (the cu129 index itself downloads at ~15 MB/s;
Chinese mirrors are also reachable). This is environmental, not a
dependency-resolution problem.

### 2. torch 2.13.0+cu129 needs nccl >= 2.29; the cu129 index has max 2.19.3

This is a pure version fact, independent of network. torch 2.13.0+cu129
calls `ncclCommResume`, a symbol introduced in nccl 2.29. The cu129
index's `nvidia-nccl-cu12` tops out at 2.19.3 (x86_64). PyPI has
nccl 2.31.2, but finding 1 makes it unreachable from the pod. Even with
perfect network, the cu129 index redirect produces a venv whose torch
cannot import: the index does not carry the full dependency closure.

**This alone is sufficient reason to revert the redirect.** The index
redirect is a claim that the index has everything torch needs; it does
not. The failure is not "the pod's network is flaky today" — it is "this
index configuration is wrong on any machine."

### 3. `uv run` on the pod is blocked by network, not dependency resolution

`uv sync` starts, resolves the lockfile, and then stalls on package
download. The resolution is correct; the delivery is impossible. The
working pod environment is `/work/tl013` (torch 2.11.0+cu129,
hand-built on Sep 2), which `pod_run.sh` puts on `PATH`. `uv run`
bypasses `PATH` and builds its own venv — but it cannot fill that venv
from the pod. **This is not the author's fault**: the fix was correct
given the information available, and the failure is environmental.

### Unmeasured

- `uv sync` wall-clock time: **not measured** — sync never completed.
- `.venv` size: **not measured** — sync never completed.

## Fix

Partial revert:

1. Keep `torch` as a direct dependency (the framework imports it
   everywhere; the transitive-via-tilelang path was always accidental).
   Relax the pin from `==2.13.0` so the lockfile is not forced onto a
   version the pod cannot run.
2. Remove the cu129 index redirect and the `[[tool.uv.index]]` entry.
   The redirect does not deliver a working torch on any machine
   (finding 2), and the pod does not use `uv run` (finding 3).
3. Regenerate `uv.lock`.

The pod continues to use `/work/tl013` via `pod_run.sh`. Mac dev and
CI resolve torch from PyPI, which works on both.

## Rule

An index redirect is a claim that the index carries the full dependency
closure. Before adding one, check that every package torch (or any
pinned dep) needs at the pinned version exists on that index — not just
the package itself. A version fact that holds on any machine is a
stronger revert reason than a network fact that holds on one.
