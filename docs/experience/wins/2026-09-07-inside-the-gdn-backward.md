# Inside the GDN backward: what a kernel port can and cannot reach — H20 sm90, 2026-09-07

> Status: **pending the measurement.** Structure and the upstream mapping are read from
> source; the seconds column is filled from the card-6 run, not before.

## Why this exists

The per-op profile put `linear_attn_chunk` at **45.300 s, 65.7%** of the attributed backward
([the table](2026-09-07-where-the-backward-goes.md)). That is a handler total. The marker on
the handler has named its replacement since day 1 —
`backend.py:1347`, `# ponytail: torch-eager backward, gdn example_chunk_delta_bwd when perf
demands` — but a handler total does not say how much of it a kernel port reaches.

One structural fact decides that, and it is readable without a card:

**`gdn_backward` recomputes the entire forward before any adjoint runs.**
`reference.py:929` runs the full chunk loop through `_gdn_chunk_fwd` — including a
`linalg.solve_triangular` per chunk — because the tape keeps no chunk intermediates. The
adjoint loop at `:955` is separate. So the 45.3 s divides into three parts with three
different levers:

| part | where | what replaces it |
|---|---|---|
| recompute of the forward chunk loop | `:923-931`, 64 chunks | **no upstream backward kernel.** A forward kernel, or a tape change that keeps the intermediates |
| the adjoint chunk loop | `:952-957`, 64 chunks | the three upstream examples below |
| prologue + epilogue | `:898-922`, `:957-994` | **nothing upstream.** conv1d taps, silu, two L2 norms, softplus, and their adjoints |

## The mapping: upstream example per sub-call

Read from each example's `prepare_output` — what it *returns* — not from its name. The three
cover disjoint outputs and together they are `_gdn_chunk_bwd`:

| upstream example | returns | our code it replaces |
|---|---|---|
| `example_chunk_o_bwd.py` | `dq, dk, dw, dg` | the `out = P s + A d` adjoint, `reference.py:637-640` |
| `example_wy_fast_bwd_split.py` | `dk, dv, dbeta, dg` | the `M = (I+L)^-1` solve adjoint, `:651-663` |
| `example_chunk_delta_bwd.py` | `dh, dh0, dv2` | the cross-chunk state scan, `:641-649` |
| `example_chunk_delta_h.py`, `example_wy_fast.py` | forward | the **recompute** at `:929`, if it stays a recompute |

Nothing upstream covers the prologue: `grep` for `conv1d`, `silu` or `l2norm` across
`examples/gdn/` returns nothing. Whatever share the prologue/epilogue row holds is outside
what these kernels can buy.

Two facts to carry into the port:

- **chunk size differs.** Ours is 16 (`_GDN_CHUNK`, `reference.py:592`), chosen for
  precision: worst rel error vs autograd is 4-12e-7 at C=16 against 1.9-4.9e-6 at C=64.
  Upstream defaults to 64 but takes `chunk_size` as a parameter, so this is a tuning value,
  not a blocker — and the parity gate is what decides it.
- the upstream README links `common/chunk_delta_h.py`, which does not exist in this checkout;
  the file is `examples/kda/chunk_delta_h_fwd.py`.

## Recompute is forced, not chosen: 164.80 GiB on a 96 GiB card

The recompute row invites "why not keep the intermediates instead". At this shape that is not
a tradeoff, it is impossible. `_gdn_chunk_fwd` returns 16 saved tensors per chunk
(`reference.py:625-627`), and at B=8, T=1280, HV=48, DK=DV=128, C=16 (80 chunks/layer):

| | f32 bytes |
|---|---:|
| one chunk's cache | 43.95 MiB |
| **one layer's caches — what `gdn_backward` holds live today** | **3.43 GiB** |
| all 48 GDN layers — what "keep the intermediates" would cost | **164.80 GiB** |
| the card | 96.00 GiB |
| the forward peak now, after the layer-wide segment ([#211](2026-09-07-a-layer-wide-checkpoint-segment.md)) | 14.90 GiB |

164.80 GiB is **1.7x the whole card**, against a forward that currently peaks at 14.90. The
single biggest entry is the incoming state `s` at 24.0 MiB/chunk — 90.00 GiB of the total — and
it is exactly the one a scan recomputes for free, so dropping it still leaves **74.80 GiB**.

The 3.43 GiB figure is the one to keep for the port: `caches` is local to one `gdn_backward`
call, so one layer's worth is live at a time and freed on return. The ×48 never coexists today,
and that is why the recompute exists.

Dims are read from the checkpoint (`text_config`: `linear_num_value_heads` 48,
`linear_key_head_dim` 128, `num_hidden_layers` 64, `full_attention_interval` 4 → 48 GDN
layers), not from the config dataclass, whose defaults are zeros.

## Method

`scripts/prof_backward_ops.py --inside-gdn` patches `reference._gdn_chunk_fwd`,
`_gdn_chunk_bwd` and `gdn_backward`. Wrapping the outer function as well as the two helpers
makes the prologue/epilogue a **measured remainder** — `gdn_backward` exclusive of its
callees — instead of a number inferred by subtraction from a total measured elsewhere.

Exclusive timing, same discipline as the registry mode, same negative control: removing the
callee subtraction puts 0.074 s on the remainder row against its own 0.020 s and fails the arm
that names it. The helpers are plain functions, not generators, so there is no drain step here.

## Results

Card 6, uncontended (8/8 at 0 MiB before launch), `--gen 1024 --group 8`, tree `cd98e3a`,
artifact `/work/gdnsplit.json`. Step 2, warm. `backward_secs` 83.613 (step 1: 80.454).

| row | raw s | share | calls | ms/call |
|---|---:|---:|---:|---:|
| `_gdn_chunk_bwd` — the adjoint | 31.268 | 55.2% | 30720 | 1.018 |
| `_gdn_chunk_fwd` — the recompute | 19.596 | 34.6% | 30720 | 0.638 |
| `gdn_backward` — prologue + epilogue remainder | 5.752 | 10.2% | 384 | 14.979 |

30720 = 384 GDN handler calls × 80 chunks a layer, which is the expected count at T=1280, C=16.

**The acceptance test 27 set does not pass on the raw numbers, and that is the finding, not a
footnote.** The rows sum to **56.616 s** against the **45.300 s** the registry-mode profile
measured for the same handler. This mode makes **61824** timed calls against that mode's 21936,
with two device syncs each, so the sync cost is inside the rows:

| | |
|---|---:|
| rows, summed | 56.616 s |
| the handler, measured without these syncs | 45.300 s |
| excess | **+11.316 s** |
| ÷ 61824 calls | **0.183 ms a call** |

Subtracting a uniform per-call cost brings the sum back to 45.300 by construction:

| row | corrected s | share |
|---|---:|---:|
| `_gdn_chunk_bwd` | 25.645 | 56.6% |
| `_gdn_chunk_fwd` | 13.973 | 30.8% |
| `gdn_backward` | 5.682 | 12.5% |

That correction is a **model** — it assumes every timed call pays the same sync — so the two
readings bracket the answer rather than one replacing the other:

- the adjoint is **55-57%** of the GDN backward
- the recompute is **31-35%**
- the prologue and epilogue are **10-13%**

The bracket is tight enough to decide the lever, which is what it was for.

### What this means for the port

Against the whole step, using the registry mode's 45.300 s for GDN and 73.775 s for the
backward:

| | s | of the GDN backward | of `backward_secs` |
|---|---:|---:|---:|
| the adjoint — the three upstream examples reach this | 25.6-31.3 | 55-57% | 35-38% |
| the recompute — no upstream *backward* kernel; `chunk_delta_h`/`wy_fast` forward would | 14.0-19.6 | 31-35% | 19-24% |
| prologue + epilogue — nothing upstream at any point | 5.7 | 10-13% | 7-8% |

**A port of the three backward examples addresses at most 57% of the GDN row**, i.e. ~37% of
the backward — not the 65.7% the one-line handler total suggests. Reaching the recompute needs
the forward kernels as well, and 10-13% is reachable by neither.

For the prologue/epilogue row, 27 asked for calls alongside seconds because elementwise
adjoints are usually launch-bound. Counted from the source blocks (`:898-922`, `:957-994`):
**45 tensor ops a call — 19 prologue, 26 epilogue — so 17280 a step** at 14.979 ms a call. The
row is 45 ops each moving f32 activations, not one expensive kernel, so its lever is fusion and
dtype, not a port. It is also the row where an f32→bf16 decision would show up.

## Not established

- **The sync correction is a model, not a measurement.** It assumes a uniform per-call cost.
  A cheaper way to settle it would be an arm that syncs only around `gdn_backward` and derives
  the two inner rows by difference — one call per layer instead of 61824.
- **The absolute seconds are not the shipped path's**, and here they are further from it than in
  the registry profile: `backward_secs` reads 83.613 against that run's 73.775, +9.838 s of
  sync. The **shares** are the quotable part, as a bracket.
- **Not a kernel profile.** A row includes python, dispatch and allocation. The recompute row
  in particular is 80 python-level chunk iterations a layer, so part of its 31-35% is loop
  overhead a kernel removes for free and part is real arithmetic — this profile does not
  separate them.
- **The port's ceiling is bounded above, not predicted.** "At most 57%" is what the adjoint
  costs today, not what it would cost after the port; a kernel that is 3x faster on that row
  buys ~37% × 2/3, and nothing here measures the kernels themselves.
- One warm step, one process, one shape, one card.

## Rule

Split a lever before sizing it. `linear_attn_chunk`'s 65.7% read as one addressable block; it
is 55-57% adjoint, 31-35% forward recompute and 10-13% elementwise pre/post, and the three have
three different fixes — a backward kernel port, a forward kernel or a tape change, and fusion.
The one-line handler total would have justified a port that reaches a third of what it looked
like it would.

When a finer probe disagrees with a coarser one, the disagreement is a quantity to measure, not
a discrepancy to explain away. 56.616 against 45.300 divided by the 39888 extra timed calls
gives 0.183 ms a call, which is a plausible cost for two device syncs — the arithmetic both
identified the cause and bounded the answer.
