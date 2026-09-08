# Porting the GDN backward to the upstream tilelang kernels — approach, not code

Written against the measured floor
([the floor probe's numbers](#the-numbers-this-rests-on)), so every claim below is either a
measurement or is marked as unmeasured. No code, no PR.

## What the port would buy, and what it would not

The four upstream kernels sum to **0.986 ms/call** at our shapes against our eager **20.3
ms/call**. Priced on 384 calls that is 0.379 s against the GDN row's 7.799 s, so
`backward_secs` 23.194 → 15.774 = **1.470x** on that BUCKET — not on the step.
The two were close enough to conflate when the bucket was 54% of the step; measured on
2026-09-07 the bucket is **26.1%** of an 85.617 s step, so 1.470x on it saves 7.153 s and
the step ceiling is **1.091x**
([the split](experience/wins/2026-09-07-the-step-is-74-percent-rollout.md)). The port did not
change; #229/#234/#240/#263 optimized the backward while this was being designed.

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

**Assertion 1 has since been run — see the section below — and it came back clean, by a route
this section did not anticipate.** It needed neither upstream's forward kernels nor the `h`/`a`/`w`
plumbing out of `_gdn_wy_core`: fla recomputes those from `A` itself, so the arm cost one card
window and no code change. What remains open is the separate **2.7e-2** figure — the upstream
tilelang kernel against upstream's *own* f32 reference, 271x the board's 1e-4 bar, the same order
as the 1.09e-2 that got the four-rounding arm rejected, and not explainable as output rounding
(arm 1). Assertion 1 does not touch it.

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

**And then overturned a second time, by the fla route.** Assertion 1 needed no plumbing either:
fla recomputes `w`/`u`/`h` from `A` internally, so the arm ran with no code change at all. This
arm's conclusion — "cannot be run as a gradient diff" — was true only of the route it assumed.
The Assertion 1 section below has the numbers.

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
So the row's total is measured and its internal parts are not. The parts come from
[where the backward goes at c128](experience/wins/2026-09-07-where-the-backward-goes-at-c128.md):
at the shipped C=128, adjoint **43.16%**, recompute 25.94%, prologue/epilogue **24.82%**, solve
6.09%. **#219's 55/35/10 bracket is superseded** — it was measured at C=16, and the chunk
increase moved the split: the prologue/epilogue runs once per layer-call either way, so its share
roughly doubled while the adjoint's fell 12 points. A port priced against 55-57% is priced
against a split the shipped tree no longer has.

## Assertion 1: run, and the conventions agree

The question this note said gated everything else — does a ported backward compute the *same*
gradient our tape computes, or a consistent variant that passes its own gradcheck and trains
differently? **Answered: the same one.**

Not against the four upstream kernels wired by hand. Against **fla 0.5.2's
`chunk_gated_delta_rule_bwd`**, which returns the whole adjoint and wires the same stages itself
(`recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` → `chunk_bwd_dv_local` →
`chunk_gated_delta_rule_bwd_dhu` → `chunk_bwd_dqkwg` → `prepare_wy_repr_bwd`). The reason is
that a hand-wiring error and a real convention mismatch produce the same symptom, and the
hand-wirer is the least reliable part of that arm. A third implementation removes the class.

`scripts/probe_gdn_assertion1.py`, card 0, artifact `/work/gdn_a1c.json`.
B=1, S=1280, NKH=16, NVH=48, DK=DV=128, **chunk 64 on all three sides**.
Versions as the run recorded them: **fla 0.5.2, Triton 3.6.0, torch 2.11.0+cu129.**

| grad | worst rel | ratio median | ratio std |
|---|---:|---:|---:|
| gstate | 2.358e-3 | 1.0 | 0.0035 |
| gbeta | 4.815e-3 | 1.0 | 0.0112 |
| gg | 6.111e-3 | 1.0 | 0.0100 |
| gq | 6.466e-3 | 1.0 | 0.0088 |
| gk | 7.023e-3 | 1.0 | 0.0082 |
| gv | **8.230e-3** | 1.0 | 0.0065 |

**Every ratio is 1.0 within 1.1%** — no sign flip, no scale factor, no decay-placement
difference. Rerun once: the six figures are bit-identical, so they are not seed or scheduling
noise.

**Where 8.23e-3 sits, stated narrowly.** It is *not* the 2.7e-2 band, so this is not the
"loose math on their side" outcome. But it does **not resolve** the 2.7e-2 either: that figure is
the upstream *tilelang* kernels against their own f32 reference, while this one is our eager f32
core against fla's Triton kernels fed bf16. Different pairs. The residual here is consistent with
those bf16 inputs. Two separate measurements, and this one says nothing about the other.

**Both sides enter and exit at the same point.** Ours drives `reference._gdn_chunk_fwd` /
`_gdn_chunk_bwd` directly — the middle of `gdn_backward` — so both start from the same `g_core`
(the gradient after the RMSNorm and z-gate adjoint) and stop at the same post-prep tensors. The
prologue and epilogue are excluded on *both* sides rather than on one.

**Not compared, and not inferable from this arm:** `gz`, `gconv1d`, `gnorm_weight` have no fla
counterpart. `ga_log` and `gdt_bias` exist only on fla's `use_gate_in_kernel` path, which also
moves its `dg` to the raw pre-softplus gate — a different quantity from the `gt` both sides share
here, so it was one or the other. Ours are pure functions of `g_gt`
(`ga_log = (g_gt * gt).sum`, `reference.py:958`), so `gg` agreeing makes them agree by
construction. Six grads, not eight.

### Three wrong verdicts, and why the numbers were never the problem

This probe printed three false conclusions before it printed a true one, and in every case the
per-row measurements were correct while the *label* derived from them was wrong. Worth recording
because the failure mode is a classifier, not an instrument.

1. **`worst rel 1424`, labelled "a real decomposition difference".** It was the probe: it fed fla
   `do=go`, the raw output gradient, where our chunk loop consumes `g_core`; and it compared
   `gdn_backward`'s dL/d(layer input) — q/k/v back through conv1d+silu+L2norm, beta pre-sigmoid —
   against fla's dL/d(post-prep). Different derivatives, so nothing would have made them agree.
2. **The ladder had no rung above `5e-2`**, so an impossible number fell into the "real
   difference" bucket by default. A verdict scale whose top rung is open-ended converts the
   author's own bug into a finding about the code under test. Fixed: `>= 1.0` now reads *the probe
   is wrong*.
3. **`8.23e-3` with every ratio at 1.0, labelled "a SCALE or SIGN convention differs".** The rung
   fired on any near-constant ratio without checking the constant differs from 1 — so perfect
   agreement was reported as a convention error. Fixed with `abs(median - 1.0) > 0.02` and an
   explicit agree rung, verified against the measured rows.

The rule: a verdict ladder needs a rung for *my own measurement being broken*, and a
"constant ratio" test is only evidence of a scale error when the constant is not 1.

## The numbers this rests on

| quantity | value | how measured |
|---|---:|---|
| upstream family, per call | 0.986 ms | 4 kernels, our shapes, 12 reps, clean cell |
| upstream family, per step | 0.379 s | × 384 calls |
| our GDN row | 7.799 s | `--inside-gdn`, instrumented — **upper bound** |
| the instrument's own cost | 4.025 s | 27.264 instrumented − 23.239 bare, same run (arm 3) |
| ratio | 20.6x | subset-vs-whole, see caveat 1 |
| BUCKET ceiling | 1.470x | 23.194 → 15.774 s |
| STEP ceiling | **1.091x** | 1.470x on a 22.372 s bucket of an 85.617 s step, 2026-09-07 |
| upstream kernel vs its own f32 reference | 2.7e-2 | dh/dh0/dv2 only; two seeds, 2.57e-2 and 2.714e-2 |
| f32-IO cell | does not compile | layout infer conflict at threads=256 (arm 1) |
| our core adjoint vs fla's | 8.23e-3 | 6 grads, ratio 1.0 ±1.1%, chunk 64 (assertion 1) |

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

Cost in files, for the plumbing a **ported backward** needs (assertion 1 itself turned out not to
need it — fla recomputes those from `A` — so this is the port's cost, not the measurement's):

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
costs two files; the rewrite costs a family of kernels and a reference rewrite. With assertion 1
now clean, the thing that would have stopped the port before it started — a convention mismatch —
is ruled out, and what remains is the 2.7e-2 and the unmeasured glue.

## The one-line recommendation

**Assertion 1 is done and clean, so the port is no longer gated on correctness-of-convention — it
is gated on the 2.7e-2 and on TP's priority.** Two open items remain, in this order: the upstream
kernels' 2.7e-2 against their own f32 reference (271x the 1e-4 bar, unexplained by rounding, and
untouched by assertion 1), and the glue cost, which only a prototype produces. A 1.470x ceiling on
the largest row in the backward justifies keeping the port on the board; it does not outrank TP.

This line has now been rewritten three times as each arm returned — "prototype assertion 1, it
needs no tape change" → "it needs two files of plumbing" → "it needed neither." Each version was
wrong about the *route* while the underlying question stayed the same, which is the argument for
running the cheap arm before pricing the expensive one.

