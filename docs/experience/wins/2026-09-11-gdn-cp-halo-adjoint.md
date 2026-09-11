# GDN context-parallel conv-halo gradient (kernel 4) — CPU, 2026-09-11

> Status: Shipped. Follow-up to the kernel-1 affine-transfer tape
> ([2026-09-11-gdn-cp-tape-end-to-end.md](2026-09-11-gdn-cp-tape-end-to-end.md)); together
> they make the CP GDN tape exact on the 27B's conv kernel 4.

## Context

The first CP-tape PR proved the affine state transfer with conv kernel 1 (zero halo).
The 27B runs kernel 4, so each chunk's depthwise prep convolution reads the predecessor
chunk's last K-1 = 3 raw qkv rows exchanged by `cp_halo`. The reverse dropped them: the
shared prep adjoint was zero-window-only, and the halo forward (an all-gather of tails)
had no reverse at all. Under zigzag cp=2 the source is not always remote — rank 1 holds
chunks 1 and 2, so chunk 2's halo is its OWN chunk-1 tail — which rules out a plain
reduce-scatter (that assumes every window comes from the other rank).

## What worked

- The shared prep save/adjoint (`_gdn_prep_save` / `_gdn_prep_backward`) now folds a
  carried `[window; qkv]` and returns the window cotangent. The reverse walks the exact
  forward index map `preact[r] = sum_tap w_tap src[width+r+tap-(K-1)]`, so the window
  rows collect their own grad; zero-window callers ignore the 9th return.
- `_gdn_backward_window` is `gdn_backward` plus the window grad (the public 11-tuple and
  its nonzero-window guard are unchanged); `gdn_cp_bwd` runs BOTH paths — token output and
  span transfer — with the real window and sums their window grads.
- `cp_halo_bwd` routes every window grad to its predecessor chunk's tail by a keyed
  all-gather + SUM keyed on the predecessor chunk id. The key (not rank order) is what
  makes rank 1's local predecessor land on its own chunk-1 tail while the remote one
  crosses ranks. It runs on both ranks with identical-shaped buffers.
- Also fixed a latent `cp_halo` forward bug: width 0 sliced `-0:` = the whole sequence;
  it now returns no rows.

Two bugs the gradcheck caught during the build: a collective deadlock from sizing the
halo buffer off `conv_windows[0]` (always None for chunk 0, so only one rank entered the
all-gather), and reading batch/feature axes off the stacked `[entries,B,T,D]` tensor.

Gradchecks (world2, real Tape + RecordingBackend, windows from a REAL `cp_halo`, floats
over a spawn Queue):

| gate | scope | worst rms-rel | red control |
|---|---|---:|---|
| `gdn_cp_halo_tape_world2.py` | conv kernel 4, all 11 leaves | 2.3e-3 | `--no-halo` q/k/v 0.46–0.76, g/beta/z/state/params unchanged |

The oracle is a single-process virtual CP that prepends each chunk's true predecessor
tail and runs chunk + span with that window. The `--no-halo` control zeroes the routed
tails while still running the symmetric gather, so it goes red ONLY on q/k/v — the exact
signature of a missing halo term.

## Rule

The reverse of a layout-keyed collective must route by the same key the forward indexes
by (chunk id), never by rank order — under zigzag a chunk's source is sometimes local. A
cross-rank adjoint gate takes its inputs from the real exchange (`cp_halo`), not a
synthetic list, and its red control corrupts the data while keeping the collective
symmetric (dropping the call just deadlocks).
