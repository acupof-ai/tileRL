# Porting the GDN backward to the upstream tilelang kernels — approach, not code

Row 49. Written against the measured floor
([the floor probe's numbers](#the-numbers-this-rests-on)), so every claim below is either a
measurement or is marked as unmeasured. No code, no PR.

## What the port would buy, and what it would not

The four upstream kernels sum to **0.986 ms/call** at our shapes against our eager **20.3
ms/call**. Priced on 384 calls that is 0.379 s against the GDN row's 7.799 s, so
`backward_secs` 23.194 → 15.774 = **1.470x** on the step.

That 1.470x is an **upper bound that will not be reached**, for three reasons that are already
known and one that is not:

1. **The comparison is a category difference, not a ratio of like things** (0a's catch). The
   0.986 ms is four kernels in isolation; the 20.3 ms is a whole eager path that also contains
   the glue between them, the head-group fold, and the norm/gate adjoint. A port replaces the
   first and keeps most of the second.
2. **The numerator is instrumented.** 7.799 s is attributed from an `--inside-gdn` arm whose
   hooks are now **measured at 4.025 s** on the whole backward (arm 3), so it is an upper bound
   on the row.
3. **The kernels compute a different decomposition** — see below. A port is a rewrite, so its
   cost is not the kernels' cost.
4. **Unmeasured:** what the glue costs once the kernels are in place. That is the number a
   prototype produces and nothing else does.

## We already ported upstream's forward family — that is the port's real starting point

Two claims an earlier draft of this note made are **false**, found by reading
`kernels_gdn.py` and `registry.py:104-109` rather than by reasoning about the examples. Both
made the port look larger than it is, so they are corrected here rather than quietly dropped.

**1. Upstream's three forward kernels are already ours.** Each of our WY-prefill kernels names
the upstream example it was ported from, in its own docstring:

| ours (`registry.py:104-109`, sm90 bf16+fp4) | upstream example |
|---|---|
| `gdn_chunk_kkt` | `example_chunk_scaled_dot_kkt.py`, `tilelang_chunk_scaled_dot_kkt_fwd` |
| `gdn_chunk_wu` | `example_wy_fast.py`, `tilelang_recompute_w_u_fwd` |
| `gdn_state_scan` | `example_chunk_delta_h.py`, `tilelang_chunk_gated_delta_rule_fwd_h` |
| `gdn_solve_tril` | `fla` `solve_tril` + `examples/kda/chunk_inter_solve_fused.py` |
| `gdn_chunk_o` | the output stage |

So "the backward family cannot be fed without also adopting the forward family" — arm 2's
verdict — is **wrong about the cost**, though right about the dependency. The forward family is
already adopted. `backend.py:1250-1257`'s `_gdn_wy_core` runs it stage for stage and its
docstring already says `saved` = "the stage outputs the adjoints take as inputs".

**2. The 48-over-16 fold is already inside the kernels, not glue.** Every one of them indexes
`kh = bh // (H // HK)` (`kernels_gdn.py:164, 283, 373, 468`) and `kernels_gdn.py:18` states the
design: "the HK key heads and the kernels index `bh // (H // HK)`: no head-repeat copy." The
earlier claim that the fold "has to live in the glue, at [B,S,48,128] → [B,S,16,128] per call"
described work that does not exist. What remains is the *gradient* direction of that indexing —
a scatter-add over 3 value heads per key head — which upstream's backward kernels do not do and
which is genuinely new.

**What is actually missing is one line.** `backend.py:1257` returns
`dict(gc=gc, a=a, w=w, u=u, h=h, v_new=v_new, chunk=chunk)` and
`backend.py:1228`'s only caller drops it on the floor:

```
core, new_state, _ = self._gdn_wy_core(qn, kn, vn, gt, bt, state)
```

`h`, `a` and `w` — the three intermediates arm 2 recorded as the wiring gap — are computed on
the sm90 forward today, in kernel layouts, and thrown away. The gap is not that we cannot
produce them; it is that nothing carries them to the tape. That is a plumbing change through
`_gdn_chunk_wy` → `linear_attn_chunk` → the tape entry, not a kernel port.

**This does not make the port cheap, and it moves where the cost is.** `_gdn_wy_core` is on the
**sm90** path only; `reference.gdn_backward` is the CPU twin and the only backward that exists.
The remaining work is the four backward kernels plus the reverse of the head-group indexing plus
the norm/gate adjoint — but the forward half, which the earlier draft priced as "seven kernels,
not four", is done.

## The decomposition delta — the substance of the work

Our `_gdn_chunk_fwd` **caches** what upstream **re-derives**. `reference.py:629`'s
`_gdn_chunk_bwd` opens by unpacking a cache built during the forward:

```
e, D, low, tri = c["e"], c["D"], c["low"], c["tri"]
KK, bp, M, W, d, QK, s = c["KK"], c["bp"], c["M"], c["W"], c["d"], c["QK"], c["s"]
```

`M` is the inverse of `(I + L)` from a triangular solve; `W` and `d` are the WY factors; `s` is
the running state per chunk. Upstream's kernels take `A` (the KK^T the `scaled_dot_kkt` kernel
produces) and `W`/`dw`/`du` as **inputs**, recomputing the equivalents inside their own tiles.
So the port is not "call a kernel where a loop was": it is **choosing which side owns those
quantities**, and the two choices have different memory and different failure modes.

- Keep our cache → the kernels' inputs are materialized from it. On the **eager/CPU** path that
  costs a transform per chunk per call; on the **sm90** path it costs nothing new, because
  `_gdn_wy_core` already produces `h`/`a`/`w` in kernel layouts and drops them.
- Drop our cache and let the kernels re-derive → the forward stops storing ~40 MB a layer
  (`reference.py:920`'s note) but the tape loses the intermediates its own adjoint uses, so
  every remaining eager piece has to be rewritten too. That is the larger change and the one
  that actually reaches 1.470x.

The recommendation and its cost in files are at the end of this note.

## The 48-over-16 head-group fold

Our config: `linear_num_value_heads = 48`, `linear_num_key_heads = 16`, so `rep = nvh // nkh` is
**3** (`reference.py:915`). The eager `gdn_backward` ends with

```
g_qn = g_qnv.reshape(b, t, nkh, rep, key_dim).sum(3)
g_kn = g_knv.reshape(b, t, nkh, rep, key_dim).sum(3)
```

a real reduction over three value heads per key head. **The forward direction of this is already
in our kernels** — see the section above; `kernels_gdn.py:18` indexes rather than copies. Every
*upstream* kernel is written with one `H` and does neither. So what a port owes is the reverse:
the adjoint of an indexed read is a scatter-add into 16 key heads from 48 value heads, and no
upstream backward kernel has it. Cheap in FLOPs, a write conflict to resolve, and **unmeasured**.

## The norm/gate adjoint and the boundary casts

`gdn_backward`'s prologue and epilogue are straight-line code, not functions: conv1d taps, silu,
two L2 norms, softplus on the way in; the norm/silu/conv adjoints and the folds on the way out.
The `--inside-gdn` arm does not wrap them, so they fall out as the row minus the two helpers, and
**no upstream kernel replaces any of it**. A port keeps all of it.

Casts: upstream's templates are `input_dtype` bf16, `accum_dtype`/`gate_dtype`/`state_dtype`
f32. Our tape carries f32 for the gradients these kernels return in bf16 (`dh`, `dv2`), so each
call adds a cast on the boundary, and — see the precision section — that cast is where a
meaningful part of the error may live.

## What the tape's op interface needs

`autograd.py:181`'s `_linear_attn_chunk` is the seam: the handler yields `(index, grad)` per
input — with grads past the 6 positional `(q,k,v,g,beta,state)` mapped onto the GDN kwargs by
`_GDN_KW` — and calls `backend.linear_attn_bwd`. So **the op interface does not change**: the
port swaps what that one backend method dispatches to. Two constraints follow from the seam
rather than from the kernels:

- **CPU twin.** The hard gate is that every op lands in the CPU cell first. The eager
  `reference.gdn_backward` already *is* that twin, so it must stay — the port adds an sm90 cell
  that overrides it, and the parity gate then compares them. This is the cheapest part of the
  work and it is also what makes the port testable at all on this machine.
- **No `torch.autograd`.** The kernels are called from the tape's handler, so nothing changes
  here; worth stating because upstream's examples are written against autograd-style
  entry points and the temptation is to import their wrappers.

## The gradcheck plan, and why its first assertion is not ours

The obvious plan — numerical gradcheck of the ported op against finite differences on the tiny
model — is **necessary and not sufficient**, and 0a named why: a gradcheck validates the port
against *our tape*. If upstream's kernel implements a slightly different adjoint (a gating
convention, a sign, a scale), the gradcheck passes and the model trains differently. So the
order is:

1. **Upstream-vs-us, on the same inputs.** Feed the same q/k/v/beta/g through
   `reference.gdn_backward` and through the four kernels wired together, and compare gradients
   directly. This is the assertion that catches a convention mismatch, and it is the one no
   amount of self-consistency testing replaces. It is also the first thing a prototype can do.
2. **Numerical gradcheck** of the ported op on the tiny model — the standing gate for any new
   backward.
3. **The parity gate** against the CPU twin at the shipped shapes.
4. **A step-level A/B**: loss curves for N steps, ported vs eager, same seed. This is what
   ultimately answers "does the error matter", and it is the only arm that measures the thing
   the bar is a proxy for.

**Assertion 1 has a measured problem already, and it is now cheaper to run than this section
first said.** The upstream kernel disagrees with upstream's *own* f32 reference by **2.7e-2** at
our shapes — 271x the board's 1e-4 bar, the same order as the 1.09e-2 that got the four-rounding
arm rejected, and not explainable as output rounding (arm 1). What changed is the cost: assertion
1 does not need upstream's forward kernels ported, because ours already are — it needs `h`, `a`
and `w` carried out of `_gdn_wy_core` instead of dropped at `backend.py:1228`.

## The three arms that were run, and what each returned

One card window (card 0, `scripts/probe_gdn_port_arms.py`, artifact `/work/gdn_arms.json`).
Each arm is isolated at its call site — an earlier version claimed the arms were independent
while arm 1's failure aborted arms 2 and 3, so the isolation is now real and was re-verified.

### Arm 1 — f32 IO: the cell does not compile

```
bf16_io: dh 2.536e-2  dh0 2.714e-2  dv2 2.222e-2   worst 2.714e-2
f32_io:  InternalError('Layout infer conflict between b_dh_fragment_2 and
         b_dh_fragment_1 in T.Parallel loop: Fragment((128, 64) -> (32,),
         replicate: 1, thread: 256, forward_thread: ...')
```

**A refusal to compile is the arm's result, not its failure.** The separator was "instantiate at
f32; if the error drops to ~1e-4 it was output rounding". It cannot be instantiated: the fragment
layouts are inferred per dtype and the f32 form conflicts at threads=256. Two consequences:

1. The bf16 error is **reconfirmed on a fresh seed at 2.714e-2**, the same order as the 2.57e-2
   the floor probe read, so it is not seed luck — **271x the 1e-4 bar**. Combined with the
   already-known fact that `dh0` is `state_dtype` (f32 on both sides today) and still reads
   2.7e-2, output rounding cannot be the explanation.
2. **"Run it at f32 and pay bandwidth" is not an available fallback.** It needs a layout fix
   inside the kernel. That is a cost a port carries, not a mitigation it can reach for.

### Arm 2 — assertion 1 cannot be run as a gradient diff

`reference.gdn_backward` ran clean and produced the full inventory (max |grad|):

| grad | shape | max | grad | shape | max |
|---|---|---:|---|---|---:|
| gq | [1,1280,2048] | 29.03 | ga_log | [48] | 157.1 |
| gv | [1,1280,6144] | 33.26 | gdt_bias | [48] | 117.5 |
| gz | [1,1280,6144] | 33.02 | gg | [1,1280,48] | 19.68 |
| gstate | [1,48,128,128] | 255.4 | gbeta | [1,1280,48] | 11.1 |

`gk` [1,1280,2048], `gconv1d` [10240,4], `gnorm_weight` [128] also returned.

The blocker is structural: upstream's `chunk_o_bwd` needs `h`, `wy_fast_bwd` needs `A` and `W`.
**No synthetic tensors were substituted** — a fabricated `h`/`A`/`W` would make the comparison
agree with itself and measure nothing, so the arm reports the gap instead of a number.

**Arm 2's verdict was then partly overturned by reading our own tree.** It concluded a port must
adopt upstream's forward family too. We already have it, and we already compute `h`, `a` and `w`
— see the first section. So assertion 1 is blocked by *plumbing*, not by three missing kernels,
and it is cheaper than this arm concluded. Recorded both ways because the arm's own output does
not contain the correction; only the source does.

### Arm 3 — the instrument's cost, measured rather than bounded

```
instrumented (--inside-gdn) 27.2644 s
bare         (--no-instrument) 23.2394 s
instrument cost                 4.025 s
```

The bare figure reconciles with the **23.194 s** that shipped in
[#240](experience/wins/2026-09-07-fp4-backward-warpgroup.md) — 0.2% apart, inside that arm's
recorded 0.493 s run-to-run spread. So the 7.799 s GDN row is now bounded by a measured
instrument cost instead of an estimated one.

**The per-op split was not captured, and that is a defect in the probe, not a property of the
system.** The filter `"gdn" in ln.lower() and ln.startswith("{")` matched only the config header
line — which happens to contain `gdn_chunk` and `gdn_forward_arm` — and none of the per-op rows.
So the row's total is measured and its internal parts are not. **#219's 55/35/10 bracket**
(adjoint 55-57%, recompute 31-35%, pre/post 10-13%) stands as the reference for the parts; no
further card window is spent re-deriving it.

## The numbers this rests on

| quantity | value | how measured |
|---|---:|---|
| upstream family, per call | 0.986 ms | 4 kernels, our shapes, 12 reps, clean cell |
| upstream family, per step | 0.379 s | × 384 calls |
| our GDN row | 7.799 s | `--inside-gdn`, instrumented — **upper bound** |
| the instrument's own cost | 4.025 s | 27.264 instrumented − 23.239 bare, same run (arm 3) |
| ratio | 20.6x | subset-vs-whole, see caveat 1 |
| step ceiling | 1.470x | 23.194 → 15.774 s |
| upstream kernel vs its own f32 reference | 2.7e-2 | dh/dh0/dv2 only; two seeds, 2.57e-2 and 2.714e-2 |
| f32-IO cell | does not compile | layout infer conflict at threads=256 (arm 3) |

**A config caveat that is part of the record**: the cell these numbers come from is the kernel
functions' *defaults* (block_DV=64, threads=256, num_stages=0). Upstream's own `main()` cell
(32/128/1) produces **all-NaN output at upstream's own advertised shape**, which invalidated an
earlier round of these numbers and made them 1.28x optimistic on the two dominant kernels. Any
port must pin the cell it validates and re-check finiteness when it changes it.

## The cache-ownership decision, with a recommendation and its cost in files

This is the decision that sizes the port, and the reading above changes the answer.

**Recommendation: keep our cache, and carry `_gdn_wy_core`'s `saved` dict to the tape.** Not
because "keep" is the conservative option — because the thing that made "keep" expensive turns
out not to exist. The transforms an earlier draft priced (materialize `h`/`A`/`W` from our cache,
per chunk, per call) are not needed on the sm90 path: those three quantities are already produced
there in kernel layouts by `gdn_state_scan`, `gdn_solve_tril` and `gdn_chunk_wu`, and then
discarded one line later.

Cost in files, for the plumbing that unblocks assertion 1:

| file | change |
|---|---|
| `backend.py:1228` | stop dropping `_gdn_wy_core`'s third return |
| `backend.py:1219` `_gdn_chunk_wy` | thread `saved` out |
| `backend.py:1160` `linear_attn_chunk` | return it on the WY branch |
| `autograd.py:181` `_linear_attn_chunk` | carry it into the tape entry |

Four files' worth of edits in two files, no kernel written, no op-interface change. That is the
step that makes assertion 1 runnable.

The **"drop our cache"** alternative is the one that reaches 1.470x, and its cost is unchanged by
this reading: the forward stops storing ~40 MB a layer (`reference.py:920`), but every eager piece
of `reference.gdn_backward` that consumes `M`/`W`/`d` has to be rewritten, and the CPU twin — the
hard gate — has to keep working. That is a rewrite of `reference.py:629-780` plus four new
backward kernels plus the reverse head-group scatter. Not this tranche.

**The two are not exclusive, and that is the argument for this order.** Plumbing the cache out
costs two files and makes the measurement possible; if assertion 1 then shows a convention
mismatch or the 2.7e-2 proves fatal, the rewrite was never started.

## The one-line recommendation

**Plumb `saved` out of `_gdn_wy_core` (two files), then run assertion 1 — before any kernel is
written.** The earlier version of this line said assertion 1 "needs no tape change", which was
wrong in both directions: it does need one, and that change is much smaller than the forward-family
port arm 2 concluded it needed. A 1.470x ceiling on the largest row in the backward justifies two
files and one card window; it does not yet justify the rewrite, and the 2.7e-2 is an unresolved
reason it might never.

