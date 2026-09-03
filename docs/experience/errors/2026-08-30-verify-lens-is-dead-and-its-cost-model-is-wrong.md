# verify_lens never runs on the shipped path, and its cost model is not this backend's — 2026-08-30

> **Superseded in part, 2026-09-03.** "Keeps everything" holds at depths 1, 2
> and 4 and nowhere else: on the 27B at depth 7 the policy trims **227 of
> 1119** rows, and 235 of 1334 overall. It binds at depth 6 and above — see
> [verify-lens-binds-where-nobody-measured-it](2026-09-03-verify-lens-binds-where-nobody-measured-it.md),
> which also finds the cost model's *shape* wrong, not only its constants.
> The cost table below stands and is what shows the shape is wrong.
>
> Status: audit finding, recorded only. No code changed — retuning a policy
> constant needs a measurement, and this host has no GPU.

## Context

`spec.verify_lens` is the DSpark §3.2.2 goodput policy: it decides how many
drafted tokens per request are worth verifying, and its docstring makes the
strong claim that "B=0 is one of the arms, so the policy never chooses to
speculate when speculating loses". Speculation loses at every point measured
([wins/2026-08-29-spec-decode-net-win.md](../wins/2026-08-29-spec-decode-net-win.md)),
which invites the question of why the policy did not switch it off.

## Two independent reasons it did not

**1. It did not run.** `engine.py:728` called
`_draft_chains(decodes, trim=not self._decode_graph_on)`, and
`_decode_graph_on = decode_graph` (`engine.py:357`), true by default on CUDA.
`_draft_chains` then returned at `if not trim: return chains` before reaching
`verify_lens`. Every captured tick drafted the full depth for every row,
regardless of confidence. The policy existed, was tested, and executed only
when graph capture was OFF — CPU, or after a capture failure.

That was not a bug on its own: a captured graph has one static width, so a
per-row trim cannot survive capture, and the entry that added it says so. The
consequence is what mattered — **the arm that could have said "do not
speculate" was the one the shipped path skipped.**

> **Update, same day.** The draft-context fix
> ([2026-08-30-draft-attention-sees-one-token.md](2026-08-30-draft-attention-sees-one-token.md))
> replaced `_draft_chains` with `_draft_step`, which calls `verify_lens`
> whenever `spec_depth > 1` regardless of capture. So it runs now — and
> reason 2 makes that a no-op, which is measured, not predicted:
>
> | spec_depth | drafts kept per tick |
> |---:|---:|
> | 1 | 1.00 of 1 |
> | 2 | 2.00 of 2 |
> | 4 | 4.00 of 4 |
>
> It keeps everything, exactly as the constants imply (below), so the change
> altered no behaviour. A policy that runs and always returns its maximum is
> not a safety net either.

**2. Its cost model does not describe these kernels**, which is why it keeps
everything. With `bias=211` and `row=0.53`, admitting one more draft raises
`theta` whenever its survival clears `0.53 * R / 211` — 0.26% at one row. The
constants are imported, and labelled as such:

```python
#: The defaults are agent-infer's H20 numbers; re-measure per target.
BIAS_MS = 211.0
ROW_MS = 0.53
```

`verify_lens` maximizes `(R + total) / (bias + row * (R + i))` — a cost linear
in the row count. Measured decode replay on this backend, from data already in
the tree
([wins/2026-08-29-m-row-gemv.md](../wins/2026-08-29-m-row-gemv.md)):

| rows | replay ms | marginal |
|---:|---:|---:|
| 1 | 10.76 | — |
| 2 | 17.54 | +6.78 |
| 3 | 27.06 | +9.52 |
| 4 | 27.58 | **+0.52** |
| 8 | 27.13 | **-0.11** |

The marginal cost of a row spans **+9.52 ms to -0.11 ms**. It is not a
constant, and no choice of `ROW_MS` makes it one: `linear_*_mma8` pads M to 8
unconditionally, so rows 5 through 8 are free and row 3 is expensive. The
shipped `bias/row` ratio is ~400; the measured one over the first two rows is
~0.6.

A cost that is flat from 4 to 8 has a different optimum in kind, not in
degree: **fill the tile to 8 rows, then stop**, rather than admit rows whose
survival clears a threshold.

## Not fixed, and why

Both fixes need a GPU run to be worth anything:

- Retuning `BIAS_MS` / `ROW_MS` replaces one unverified constant with another —
  the numbers above are decode replays, not verify ticks, and the entry that
  produced them warns that mixing readings from two configurations is how this
  feature got a 1.9x optimistic estimate once already.
- Replacing the linear model with a tile-aware one changes which drafts are
  kept, and the only thing that says whether that is better is measured
  goodput.

Recorded so the next person does not read `verify_lens` as an active safety
net. It is neither active nor, at these constants, correct.

## Rule

A policy that never executes cannot be blamed for a bad outcome, and cannot be
credited for a good one. Check that the branch runs before reasoning about
what it decided.

Corollary: a constant carrying a "re-measure per target" comment is an
unpaid debt, not a default. This one was 400x off in ratio and nobody noticed,
because the code holding it does not run.
