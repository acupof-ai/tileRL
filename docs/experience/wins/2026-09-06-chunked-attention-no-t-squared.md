# Chunked attention on the training path: no T×T score matrix — cpu, 2026-09-06

> Status: Shipped (cpu) · pending-remote (sm90 step time and gen-4096 peak)

## Context

`dense_attention` builds `att [B,H,Tq,Tk]` in f32. At T=4352 with 24 heads that
is 1.69 GiB, and it is the only allocation in a training step that grows with
T². Backward is 54.4% of a GRPO step at group 8 (71.5 s of 131.6,
[#192](2026-09-06-one-grpo-step-is-54-percent-backward.md)), so the score matrix is both the
memory ceiling and inside the term that dominates wall clock.

**The "27.1 GiB over 16 layers" figure this work started from is wrong.** The
tape saves q/k/v — `autograd.py:149` reads `args[0:3]` — and never `att`, which
is a local of `dense_attention` freed on return, so layer depth does not
accumulate. Measured peak *simultaneously live* `[*,*,T,T]` bytes across 16
sequential layers, by refcount death through a `TorchDispatchMode`: **3.00x** one
att tensor in forward, **6.00x** in backward. Forward holds `att`, the mask sum
and `p`; backward adds `gatt`, `gp` and the `(gatt*p)` temporary. So the real
ceiling at T=4352, H=24, B=1 is **5.08 GiB forward / 10.16 GiB backward**, not
27.1. The correction to #192's entry is in PR #195, not yet on main — that entry
still reads `× 16 layers = 27.1 GiB` at the time of writing.

## What Worked

`chunked_attention` / `chunked_attention_bwd`: the same result computed one
`[C,C]` score tile at a time, C=64. Identical signatures, so `Backend.attention`
/ `attention_bwd` and `ReferenceBackend` swap underneath and `autograd.py` /
`model.py` are untouched. Dense keeps its body as the parity oracle.

Peak live bytes of **every** intermediate, T=1024, B=1 H=4 D=8:

| | dense | chunked | ratio |
|---|---:|---:|---:|
| forward | 80.12 MiB | 0.58 MiB | 138x |
| backward | 128.50 MiB | 1.16 MiB | 111x |

The largest single tensor becomes input-shaped `(1,1024,4,8)` against dense's
`(4,1024,1024)`, and stays byte-identical when T doubles — that, not the ratio,
is the property the gate asserts.

Three mechanisms carry it:

- **Online softmax** over K tiles, rescaling the accumulator and the row sum by
  `exp(m_old - m_new)`.
- **Backward recomputes P and stores nothing.** Pass 1 is the forward's loop
  kept for `m`/`l`/`O` only, to form `delta = (grad*O).sum(-1)`; pass 2
  recomputes each tile's scores and forms `P = exp(s-m)/l` directly, needing no
  rescale because m/l are already final. 2x the score FLOPs, the usual flash
  trade. Returning m/l from the forward would change `attention_bwd`'s
  signature for a cost nothing has measured.
- **GQA without expanding K/V.** The group is its own einsum axis, so key grads
  accumulate straight into `[B,Tk,Hkv,D]`; dense expands to Hq then folds.

Causal skipping is per tile pair from **absolute positions**, so CP's zigzag
`k_pos` is honoured rather than assumed contiguous — and computed on device,
reduced once: a per-tile `int(k_pos.min())` cost **5 host syncs per train step**,
caught by `test_train_step_does_not_sync_per_parameter` reading 7 against a
budget of 2. Now 0 syncs at T=128/256/512.

Parity against dense, max abs rel:

| shape | out | gq | gk | gv |
|---|---:|---:|---:|---:|
| B=1 T=2 H=4 (one tile) | 8.4e-08 | 4.5e-07 | 4.4e-07 | 0.0 |
| B=2 T=200 Hq=8 Hkv=2 (GQA) | 6.2e-07 | 5.4e-07 | 6.0e-07 | 5.0e-07 |
| B=1 T=150 Hq=6 Hkv=3 (odd tail) | 2.3e-07 | 3.2e-07 | 4.9e-07 | 3.6e-07 |

## Controls

Each mutation applied alone; out rel against dense at B=2 T=200 GQA.

| reverted | goes red |
|---|---|
| the `acc *= r` rescale | parity, 1.80e+00 |
| the `l *= r` rescale | parity, 5.61e-01 |
| the causal mask | parity, 1.26e+00 |
| the on-device tile reach (per-tile `int()` back) | the sync gate, `fwd T=128: 8 host syncs` |

The two rescale arms are unreachable at one tile by construction, which the T=2
row documents by staying green for them at 8.4e-08.

Two defects the gates caught that reading the diff did not:

- `nan_to_num_` in-place broke the gradcheck (`output 0 of Exp, is at version 1`)
  — the harness runs this forward under `torch.autograd`.
- The host syncs above. `-k attention` deselects the test that found them, so a
  green filtered run said nothing about it.

## Not established

- **Every number here is the CPU target.** No sm90 kernel; the arch cell is the
  follow-up and C=64 is untuned (`# ponytail` names the sm90 cell as where it
  gets measured).
- **No step-time number.** 2x score FLOPs against a 138x memory drop is a trade
  whose sign on wall clock is unmeasured on GPU; on CPU the suite is 3:21 either
  way, which measures nothing about the kernel.
- **The claim this unblocks is untested**: whether `gen 4096` at group 8 fits on
  one card. That needs the pod.

## Rule

A per-call tensor size times a layer count is not a live-bytes figure. What the
tape saves is a property of the handler — here `args[0:3]`, so 13 of the 16
score matrices never coexist and the true peak is 3x/6x one of them, not 16x.
Read the handler before multiplying.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | (this PR) | mac m-series | cpu | tiny | n/a | n/a | n/a |
| pending-remote | | H20 card 6 | cuda sm90 | 27B | | | |

The cpu row carries no timing on purpose: this entry's measurement is memory and
parity, and the sm90 row is where step time and the gen-4096 peak land.

Raw artifacts: `tests/test_ops_parity.py` (11 gates incl. 4 controls); memory and
sync figures reproduce from
`test_chunked_attention_allocates_no_score_matrix` and
`test_chunked_attention_does_not_sync_per_tile`.
