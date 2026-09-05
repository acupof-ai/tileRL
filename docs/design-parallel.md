# Parallel training: TP first, then CP/SP

Design only, no code. Reviewed before implementation.

## What already exists

Not greenfield, and the design has to say what it is *adding*:

| shipped | where |
|---|---|
| `Backend.all_reduce` / `all_gather` / `tp_fork` | `backend.py:153`, `:164`, `:176` |
| weight sharding, column/row split tables, fp4/fp8 alignment checks | `tensor_parallel.py` |
| `pad_vocab`, `kv_replicas`, `tp_config` | `tensor_parallel.py:59`, `:65`, `:79` |
| row-parallel all-reduce in the forward | `model.py:191` (`_add_via`) |
| data-parallel engines, one per card | `parallel.py` |
| NCCL cost: **21.5 µs per call, flat 20 KB → 1.3 MB** | CHANGELOG 2026-08-30 |

**The backward landed in #115.** `_BWD` (`autograd.py:245`) now registers
`all_reduce`, `all_gather` and `tp_fork`, gated by `tests/tp_world2.py` on two
gloo ranks. The sharded cross-entropy followed in #119. **What remains of this
document is the mesh** — nothing yet selects a `(dp, tp, cp)` layout — and CP/SP
in section (e).

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
| vocab-parallel `lm_head` | **sharded CE**, three scalar `all_reduce` per row | gradient stays sharded |
| GDN scan, CP | `all_gather` of local `(A, B)` | same scan reversed |
| ring attention, CP | `send`/`recv` ring | reversed ring |
| SP norm/embedding | `all_gather` in, `reduce_scatter` out | mirrored |
| dp gradients | none | `all_reduce(grad)` once per step |

The pattern is that column and row are **duals**: a forward all-reduce means no
backward one, and vice versa. Getting that backwards costs a factor of `world` in
the gradient and is silent — which is what the gate below is for.

**The `lm_head` row is the one that must not follow the pattern.** All-gathering
logits materialises `[B, T, 248320]` in f32: one row is **0.947 MiB** at this
vocab, so **1.89 GiB per rank per step at B=8, T=256** (3.79 GiB at T=512), on top
of the weights and the tape. (Not the 08-30 OOM's 8.5 GiB, which this document
previously cited — that was three `(3072, 248320)` tensors from `lm_head` over
every position at B=32, a different failure.) The sharded cross-entropy never
forms them: each rank takes a local row max and a local sum-exp, and **three**
scalar `all_reduce`s per row combine them — max, sum-exp, and one selecting the
target column, which lives on exactly one rank and contributes zero elsewhere.
The gradient is produced already sharded. Shipped in #119.

On the tape it took **three** entries, not two: `all_reduce` → identity,
`all_gather` → keep this rank's chunk (a `reduce_scatter` was never needed, since
nothing sums across ranks on that path), and **`tp_fork`** — identity forward,
all-reduce backward — which this document did not anticipate. `tp_fork` is where
the column-parallel dX sum lives, and it turned out to be needed for replicated
*weights* over sharded heads (`q_norm`/`k_norm`/`gdn_norm`) as well as for
activations. The tape is hand-written, so a collective is an op like any other; no
autograd hooks.

## (d) Gates

CPU, world=2, gloo, tiny model — runs here, no card:

1. **TP-2 loss and gradients equal TP-1**, on the same seed and batch. Not loss
   alone: a missing backward all-reduce leaves the loss right and the gradient
   wrong by a factor of `world`, and the loss recovers by the next step. SHIPPED
   (`tests/tp_world2.py`): all 54 tensors compared by shipping the world=1
   gradients through `shard_params`, since comparing only the replicated ones
   passes with every sharded gradient wrong; both head layouts, because `tiny()`
   ties its head and the vocab-parallel branch is the one the 27B takes. The
   1e-3 here was a guess — the measured bf16 spread is 1.4e-4, and the shipped
   tolerance is rtol 2e-3.
2. **CP-2 the same**, with a sequence long enough to cross the chunk boundary —
   a CP test on one chunk exercises nothing.
3. **Negative control for each**: delete the backward collective and the test must
   fail. Without it, "TP-2 matches TP-1" also passes for `world` silently 1.
   `--no-fork` does this and fails on both head layouts. For the sharded CE the
   control is a resource one — `--gather` is numerically correct and must fail
   the memory assertion — and that assertion is a shape, not a byte count:
   `tracemalloc` reports 80 bytes for a 4 MB torch tensor, so bytes cannot be
   asserted on the CPU target.
4. 27B numbers `pending-remote`.

## (e) Order

1. **TP training.** DONE except the mesh: three `_BWD` entries (#115) and the
   sharded CE (#119), both gated. The mesh is what still blocks full-parameter
   RL, which is the P3 dependency.
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
