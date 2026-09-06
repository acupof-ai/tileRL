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

## Method

`scripts/prof_backward_ops.py --inside-gdn` patches `reference._gdn_chunk_fwd`,
`_gdn_chunk_bwd` and `gdn_backward`. Wrapping the outer function as well as the two helpers
makes the prologue/epilogue a **measured remainder** — `gdn_backward` exclusive of its
callees — instead of a number inferred by subtraction from a total measured elsewhere.

Exclusive timing, same discipline as the registry mode, same negative control: removing the
callee subtraction puts 0.074 s on the remainder row against its own 0.020 s and fails the arm
that names it. The helpers are plain functions, not generators, so there is no drain step here.

## Results

(pending — card 6, `--gen 1024 --group 8`, tree `cd98e3a`)

Acceptance: the rows must sum to the 45.3 s within the sync overhead, or the probe is not
measuring the handler.

## Not established

- (pending the run)
