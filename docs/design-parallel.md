# Parallel training: TP first, then CP/SP

Design only, no code. Reviewed before implementation.

## What already exists

Not greenfield, and the design has to say what it is *adding*:

| shipped | where |
|---|---|
| `Backend.all_reduce` / `all_gather` over `torch.distributed` | `backend.py:154-173` |
| weight sharding, column/row split tables, fp4/fp8 alignment checks | `tensor_parallel.py` |
| `tp_config`, `pad_vocab`, `kv_replicas` | `tensor_parallel.py:59-101` |
| row-parallel all-reduce in the forward | `model.py:183` (`_add_via`) |
| data-parallel engines, one per card | `parallel.py` |
| NCCL cost: **21.5 µs per call, flat 20 KB → 1.3 MB** | CHANGELOG 2026-08-30 |

**The gap is the backward.** `_BWD` (`autograd.py:224`) registers no collective,
so every shipped collective is inference-only. Nothing in the tree trains sharded.
That, plus a mesh, is this work.

## (b) Topology

One `Mesh` over `(dp, tp, cp)`, sizes multiplying to `world`. Rank layout is
**cp fastest, then tp, then dp**:

```
rank = ((dp_i * tp) + tp_i) * cp + cp_i
```

so a tp group is contiguous and lands inside one node, which is where NVLink is.
CP ranks pass a scan state per layer and go next; dp ranks talk once per step and
go outermost.

Three layouts for a 27B on 8 H20 (22.8 GB weights, 24 attention / 4 KV heads, 48
GDN value heads):

| | shards | KV heads/card | weights/card | comment |
|---|---|---:|---:|---|
| `tp=8` | 8 | 4/8 → replicated ×2 | 2.85 GB | `kv_replicas` already handles 4 heads over 8 ranks |
| `tp=4, dp=2` | 4 | 1 | 5.7 GB | two independent rollout streams; the RL default if weights fit |
| `cp=8` | 1 | 4 | 22.8 GB | sequence split; only worth it past ~64K where activations dominate |

`tp=8` is the only one that fits a *full-parameter* step, which is why it is first:
LoRA fits on one card, full-param does not.

## (c) Collectives seam

Collectives live in **`Backend`**, in the kernel package — backend isolation says
only `tilerl_kernels` touches `torch.distributed`, and `all_reduce`/`all_gather`
are already there. The mesh lives in `tilerl` and calls through `Backend`. **No
`src/tilerl/comm.py`.** NCCL on CUDA, **gloo on CPU** so the parity gate runs on
this machine — gloo has `all_reduce`/`all_gather`/`reduce_scatter` and
`send`/`recv`, which is the whole surface.

```
all_reduce(x, group)        reduce_scatter(x, group)
all_gather(x, dim, group)   send(x, to) / recv(x, from)     # ring
```

What each layer needs, **forward → backward**, the row that does not exist yet:

| layer | forward | backward |
|---|---|---|
| column-parallel linear (`q/k/v/gate/up`) | none | `all_reduce(dX)` |
| row-parallel linear (`o/down`) | `all_reduce(Y)` | none |
| vocab-parallel `lm_head` | **sharded CE**, two scalar `all_reduce` per row | gradient stays sharded |
| GDN scan, CP | `all_gather` of local `(A, B)` | same scan reversed |
| ring attention, CP | `send`/`recv` ring | reversed ring |
| SP norm/embedding | `all_gather` in, `reduce_scatter` out | mirrored |
| dp gradients | none | `all_reduce(grad)` once per step |

The pattern is that column and row are **duals**: a forward all-reduce means no
backward one, and vice versa. Getting that backwards costs a factor of `world` in
the gradient and is silent — which is what the gate below is for.

**The `lm_head` row is the one that must not follow the pattern.** All-gathering
logits materialises `[B, T, 248320]` in f32 — the **8.5 GiB** that failed on
08-30. The sharded cross-entropy never forms them: each rank takes a local row max
and a local sum-exp, two scalar `all_reduce`s per row combine them, and the
gradient is produced already sharded. So the new `_BWD` entry is for **that op**,
not for a `reduce_scatter` of `dlogits` that would have to exist first.

On the tape, the rest is two `_BWD` entries (`all_reduce` → identity, `all_gather`
→ `reduce_scatter`) plus recording the existing calls. The tape is hand-written, so
a collective is an op like any other; no autograd hooks.

## (d) Gates

CPU, world=2, gloo, tiny model — runs here, no card:

1. **TP-2 loss and gradients equal TP-1** to 1e-3, on the same seed and batch.
   Not loss alone: a missing backward all-reduce leaves the loss right and the
   gradient wrong by a factor of 2, and the loss recovers by the next step.
2. **CP-2 the same**, with a sequence long enough to cross the chunk boundary —
   a CP test on one chunk exercises nothing.
3. **Negative control for each**: delete the backward collective and the test must
   fail. Without it, "TP-2 matches TP-1" also passes for `world` silently 1.
4. 27B numbers `pending-remote`.

## (e) Order

1. **TP training.** Two `_BWD` entries, the mesh, the gates. Unblocks
   full-parameter RL, which is the P3 dependency.
2. **CP.** GDN as the scan `S_i = A_i S_{i-1} + B_i` (composes, so one all-gather
   fixes the incoming state); ring attention for full attention, whose merge is
   the split-KV combine already shipped.
3. **SP is not a third mechanism** — it is CP applied to the norm and embedding
   traffic that TP leaves replicated. It lands as part of CP, not before it, and
   it is worth naming only because it is where the last replicated activation
   memory goes.

## Two risks worth stating now

- **TP forfeits the decode graph** until `Backend.all_reduce` is capturable
  (`tensor_parallel.py:21` already says so). The graph is worth **2.16× on the RL
  step** (measured 2026-09-05 on card 6: 73.62 → 34.09 s, n=10 pooled over both
  arm orders — `wins/2026-09-05-recapture-after-update.md`), and it is now the
  default there. So TP-8 that loses it starts 2× behind and has to win that back
  before it is a gain. This is the single biggest number in this document and it
  argues for `tp=4, dp=2` over `tp=8` wherever the weights fit.
- **The GDN scan is the one novel kernel.** It does not touch the 27B until the
  tiny gradcheck passes.
