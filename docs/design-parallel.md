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
| GDN scan, CP | log₂ prefix scan of `(A, B)` | same scan reversed |
| **full attention, CP** | `all_gather` of K/V | **`reduce_scatter(dK, dV)`** — not a slice |
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

## (d2) CP: ring or all-gather, decided on the arithmetic

The attention CP op is the first one needing a CPU twin, so the collective is
named here before any code. Both candidates move **identical bytes** — computed
for the 27B (16 full-attn layers, 4 KV heads, head_dim 256, bf16 K and V):

| | cp=2 | cp=4 | cp=8 |
|---|---:|---:|---:|
| bytes/step, either scheme (B=8, T=256) | 64 MiB | 96 MiB | 112 MiB |
| **ring** calls/step | 16 | 48 | **112** |
| **all-gather** calls/step | 16 | 16 | **16** |
| ring latency floor @ 21.5 µs/call | 344 µs | 1032 µs | **2408 µs** |
| all-gather latency floor | 344 µs | 344 µs | **344 µs** |

**Both columns double once the backward is counted.** The backward of the K/V
gather is a `reduce_scatter`, not a slice (measured, see below), so every step
moves the same bytes twice and issues the same calls twice: at cp=8, 224 MiB and
ring 4816 µs against all-gather 688 µs. The **ratio is unchanged** — the doubling
is common to both schemes — so the decision below stands on the same 7x.

Bytes are the same because each scheme moves every remote chunk to every rank
exactly once. What differs is the **call count**: ring is `cp-1` sequential
send/recv per layer, all-gather is one. Against the measured 21.5 µs NCCL floor
(flat from 20 KB to 1.3 MB, CHANGELOG 08-30) that is a 7x latency difference at
cp=8, and the chunks here are inside the flat region, so the floor *is* the cost.

**That table assumes both are fully exposed, which is unfair to ring** — ring's
whole point is that hop `i+1` overlaps block `i`'s attention compute, so its
exposed cost is `max(compute, hop)` per block, not the sum. The overlap only
helps if there is compute to hide behind, so: at B=8, T=256, cp=8 each block is
**32 tokens**, and one full-attn layer's attention over it is
`2·B·H·blk²·D·2` = **0.201 GFLOP**.

| effective throughput | per-block attention | hop floor |
|---|---:|---:|
| 100 TFLOP/s (conservative) | **2.0 µs** | 21.5 µs |
| 300 TFLOP/s | 0.7 µs | 21.5 µs |
| 989 TFLOP/s (H20 bf16 peak) | 0.2 µs | 21.5 µs |

Compute is **10–100x below the hop floor**, so `max(compute, hop) ≈ hop` and
overlap recovers almost nothing at these shapes. The 2408 µs stands. This flips
only when a block is large enough for its attention to exceed 21.5 µs — around
T/cp ≈ 400+ tokens at 100 TFLOP/s, i.e. the long-sequence regime where ring also
wins on memory. Same threshold, twice.

**What this decision rests on, stated plainly.** These are FLOP-model numbers,
not a measurement: there is no CP attention kernel to bench, so nobody has timed
the quantity that actually decides it — the **exposed** comms cost, how long a
layer stalls with ring's hops overlapping block compute. The choice above is made
on unoverlapped floor arithmetic plus a two-orders-of-magnitude margin, and it is
**revisitable the moment that number exists**. If a real ring implementation
overlaps better than the model says, or the attention kernel is far off 100
TFLOP/s effective at 32-token blocks, this flips. Treat it as the default to
build first, not as settled.

What ring buys instead is memory — it never holds the whole KV:

| | B=8 T=256 | B=8 T=2048 | B=1 T=65536 | B=1 T=131072 |
|---|---:|---:|---:|---:|
| whole KV per rank, all-gathered | 0.12 GiB | 1.00 GiB | 4.00 GiB | **8.00 GiB** |
| + dK/dV f32, one layer live | 0.14 GiB | 1.12 GiB | 4.50 GiB | **9.00 GiB** |

The backward adds a flat **+12.5%**, not another factor: the gathered K/V are
retained for all 16 layers, while dK and dV exist for one layer at a time. It
moves the crossover by that much and no more — a 4 GiB budget at B=1 holds
**58254** tokens rather than 65536.

**So: all-gather for the RL path, ring only past ~58K** — on floor arithmetic,
pending an exposed-cost measurement (see below). At the shapes RL
actually runs (B=8, T=256–2048) the gathered KV is 0.14–1.12 GiB against a 96 GB
card and all-gather is 7x cheaper in latency at cp=8. Ring's memory advantage
only pays once the whole KV stops being free, which is the ~64K threshold
section (b) gives for CP being worth doing at all, pulled in to ~58K by the
backward's dK/dV. Implementing ring first would be optimizing the case CP is not
yet used for.

This makes the first CP op an `all_gather` of K and V per full-attn layer —
already on the tape (#115). Its backward is a **`reduce_scatter`**, not a slice.
The slice rule holds for the vocab gather, where each rank owns a distinct slice
of the output; it fails here because every rank's queries read every rank's keys,
so each rank's dK covers the whole sequence as a partial sum. Measured against a
dense single-rank reference at cp=4: slicing is wrong by **1.03 of a 1.78 peak
(58% of full scale)**, summing matches to **2.4e-07**. dQ stays local. Ring stays
a `# ponytail:` note on it.

**Chunks are assigned zigzag, not contiguously.** Under a causal mask rank r's
queries score against chunks 0..r, so contiguous chunking leaves the last rank
doing `(cp+1)/2` times the mean work while the gather barrier waits on it. Rank
r takes chunks `r` and `2cp-1-r`, so every rank's pair sums the same. Measured at
cp=4, T=32, scored q·k pairs per rank: contiguous **[36, 100, 164, 228]** (6.3x
max/min), zigzag **[132, 132, 132, 132]** (1.0x). The gathered chunks therefore
arrive in **rank order, not sequence order**, so the mask is built from absolute
positions passed alongside the tensors — never inferred from an index.


### The GDN scan: a log₂ prefix scan, not an all-gather

The state is `B × 48 × 128 × 128` f32 = **24.0 MiB per layer**, 1.12 GiB across
48 layers at B=8. The recurrence composes, so a rank holding a later chunk never
needs its predecessor's state — only the composed `(A, B)` of everything before
it, and composition `(A₂A₁, A₂B₁ + B₂)` is associative, so the prefixes come from
a scan.

**`A` is a `DK × DK` operator, not the decay scalar.** Reading `_gdn_chunk_fwd`:
`d = U − W s`, so `s_next = exp(glast)·s + Rᵀd = (exp(glast)I − RᵀW)·s + RᵀU`.
The state's own influence runs through `d`, and folding it in is what makes the
recurrence affine. The natural misreading — decay the state, add the chunk's
contribution — is **wrong by 23%** (measured) and still produces a plausible
loss. `tests/gdn_cp_scan.py --decay-a` is that misreading as a control.

**Three schemes, priced with the compute column** (48 GDN layers, HV=48,
DK=DV=128, f32, 400 GB/s assumed, 21.5 µs measured floor, whole step in ms):

| | cp=2 | cp=4 | cp=8 |
|---|---:|---:|---:|
| sequential state pass | 26.0 | 32.1 | 44.2 |
| `(A,B)` all-gather | 34.8 | 50.6 | 94.8 |
| **`(A,B)` log₂ scan** | **28.7** | **32.5** | **40.4** |

*T=2048, B=8, 300 TFLOP/s effective.* The sequential pass is not cheap because
of its bytes — it is cheap on bytes — but rank r idles until rank r−1 finishes,
so **every rank pays the full-sequence compute** and the step gets *worse* with
cp (26.0 → 44.2). That is CP buying nothing on 48 of the 27B's 64 layers. The
all-gather loses on bytes instead: `(A, B)` is 2× the state per chunk and zigzag
gives `2cp−1` remote chunks. The scan needs only prefixes: `⌈log₂ cp⌉` steps of
2 states each — 3 steps at cp=8 against 15 payloads.

**At the shapes CP is actually for, the margin is not close:**

| T, B | cp=2 | cp=8 |
|---|---|---|
| 2048, 8 | seq by 1.10× | scan by 1.09× |
| 65536, 1 | scan by 1.37× | **scan by 4.69×** |
| 131072, 1 | scan by 1.37× | **scan by 5.06×** |

Sequential's compute term is the whole sequence on every rank while the scan's
divides by cp, and per-layer comm barely moves, so the gap opens with T. The one
cell where sequential wins is **T=2048, cp=2 at 300 TFLOP/s** — below 100
TFLOP/s effective the scan wins there too (1.16×). That is the only
throughput-sensitive cell; everywhere else the ordering is stable.

**Cost line:** building `(A, B)` and applying the prefix correction is **+45%
FLOPs** on the layer (+64.4 of 143.9 GFLOP at T=2048, B=8). It parallelizes, so
it divides by cp, which is why the scan's compute column is not `base/cp`.

These are FLOP-model numbers with an assumed 400 GB/s. No CP kernel exists, so
the exposed cost is unmeasured — the same caveat as (d2), and the same
revisit-when-measured standing.

## (e) Order

1. **TP training.** DONE except the mesh: three `_BWD` entries (#115) and the
   sharded CE (#119), both gated. The mesh is what still blocks full-parameter
   RL, which is the P3 dependency.
2. **CP.** Attention first: the K/V all-gather with its `reduce_scatter` backward
   and zigzag chunking, gated by `tests/cp_world2.py`. **GDN second, and CP is
   unusable until it lands** — 48 of the 27B's 64 layers are GDN, and `_gdn`
   refuses `cp>1` rather than start a chunk from a zero state. The scan
   `S_i = A_i S_{i-1} + B_i` composes, so a **log₂ prefix scan** of the per-chunk
   `(A, B)` fixes the incoming state — not an all-gather, which loses to a plain
   sequential state pass at every cp measured, and not the sequential pass
   either, which makes every rank pay the full-sequence compute. Attention is not
   ring, per the arithmetic in (d2): same bytes, 7x fewer calls at cp=8, and
   ring's memory advantage does not pay below ~58K.
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
