# GDN context-parallel world2 gates green on two H20 cards — 2026-09-11

> Status: **measured on CUDA**, cards 0+1 (world 2), commit
> `a08e546ad836246de272d599e557d56d0aadbb87`. These are the card confirmations
> of the gates that were green only on CPU gloo; the future P6 exit (8 cards,
> 32K gradients to 1e-3, 256K fwd+bwd) is a separate run.

## Context

The GDN CP tape (affine prefix-scan transfer + its reverse) and the cross-rank
conv-halo exchange shipped with world2 gates runnable on CPU gloo, but the
runbook ([PENDING-REMOTE-CARDS](../PENDING-REMOTE-CARDS.md), step 5) held their
CUDA confirmation until cards returned. A gloo pass does not exercise the NCCL
collectives, the sm90 cells, or two real devices' state exchange.

## What ran

`CUDA_VISIBLE_DEVICES=0,1`, system `/work/tl013` python with
`TILERL_TARGET=cuda PYTHONPATH=src:packages/tilerl-kernels/src`, tree
`/work/tilerl-cc` at `a08e546a` (`.synced_dirty=0`, so rows/gates run clean).
Unique per-gate gloo ports let them run back to back in one detached job.

| gate | card output (verbatim tail) | rc |
|---|---|---:|
| `gdn_world2.py` | `gdn cp=2 zigzag: every chunk from its scanned prefix matches the sequential scan` | 0 |
| `gdn_cp_gradcheck_world2.py` | `affine scan reverse matches central differences` | 0 |
| `gdn_cp_tape_world2.py` | `gdn_cp world2 tape matches global central differences worst 4.13e-04` | 0 |
| `gdn_halo_world2.py` | `gdn halo cp=2: every chunk with its left context matches the sequential run` | 0 |
| `cp_world2.py` | `cp: a split sequence matches the unsplit forward and backward` | 0 |

The tape's worst central-difference error is **4.13e-4** on CUDA (the CPU run
quoted ~4e-4 in the runbook), so the cross-rank reverse matches the single-rank
global-loss oracle at the same order through NCCL, not just gloo.

### The inverted control

`gdn_halo_world2.py --no-halo` is *supposed* to fail the rel check — a vacuous
gate would exit 1. It prints `no-halo control: correctly FAILED` and exits **0**
(the harness inverts the exit code for the control). Verified on the cards:

```
no-halo control: correctly FAILED
process-rc=0
```

## Rule

A CP/collective gate green only on gloo is a hypothesis about the CUDA path: run
the same tape through two real devices and compare to the single-process
sequential oracle, and keep the negative control whose exit code is inverted so
a pass that never tested the halo cannot masquerade.

## Results

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | H20 cards 0+1 | cuda sm90, world 2 | 5 gates rc=0; tape worst 4.13e-4; --no-halo correctly FAILED |
