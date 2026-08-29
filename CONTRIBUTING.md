# Contributing to tileRL

## Layout

A uv workspace with two distributions, split along the boundary the codebase
already enforces:

```
packages/tilerl-kernels/src/tilerl_kernels/   TileLang kernels + the Backend seam
src/tilerl/                                   engine, model, server, training
```

`tilerl_kernels` is the only place that imports TileLang or calls torch beyond
the tensor container type. Everything in `tilerl` is backend-neutral and talks
to kernels through `Backend`. If a change makes something above the seam import
TileLang, the split is being broken, not extended.

```bash
uv sync            # both packages editable, plus every dependency a gate touches
uv run pytest      # the deterministic CPU suite CI runs
uv run ruff check
```

## Gates

A change is not done until it clears the gates its area owns.

**Every runtime change lands a bench entry.** A dated file under
`docs/experience/wins/` — or `errors/` when the measurement rejects the change,
which happens often and is worth as much. No entry, not shipped. Exempt: docs,
agent files, dev-only tooling; say so in the commit body. Can't run a target
locally (no GPU on a Mac) → stub `pending-remote`, never a silent skip.

**Correctness parity for every new op.** TileLang on the CPU target against
`tilerl_kernels.reference`, `allclose(rtol=1e-2)`, on the tiny model. The
reference is the executable spec; when they disagree, the reference is right
until proven otherwise.

**Numerical gates run at the model's real magnitude.** A parity fixture at
scale 0.1 passes things that are 26% wrong at scale 1.0 — that is how one
rejected kernel got its accuracy verdict backwards. See
`test_gdn_chunk_fused_parity_full_scale`.

**Gradient check for every new backward.** The tape is hand-written, so silent
gradient errors are the default failure mode, not an exotic one.

**Accuracy is a gate, not a report.** `tilerl bench --suite accuracy` scores a
fixed greedy MMLU slice. Every other suite measures speed; without this one a
change that breaks the logits passes all of them.

## Measuring

Three rules, each of which cost a day to learn:

1. **Compare against the configuration that ships.** Speculative decoding read
   as a 1.14x win against the eager path while the shipped decode is
   graph-captured, where it is 0.43-0.76x. The number was real; the baseline
   was not.
2. **Never compare across scripts.** Different prompts, warmups and settle
   behaviour hide inside two tools that look like they measure the same thing.
   The bench suites now measure both arms themselves for this reason.
3. **Measure the upper bound before building.** N processes bound any
   in-process data-parallel wrapper; a kernel's grid bounds any schedule
   change; `ncu` names a limiter in one run. Each costs minutes and has
   repeatedly replaced an hour of implementation.

## TileLang

Kernel bodies are TVMScript-parsed, not executed Python. Four traps have
silently produced wrong numbers here, all the same shape — a Python value
leaking into traced control flow:

| written | what it did |
|---|---|
| `expr and vs == 0` | `bool()` on a symbolic operand is always true |
| `T.serial(K // VB)` | symbolic bound makes a shared index dynamic, breaking the layout |
| `T.serial(KSP - 1)` | zero-trip loop when the count folds to 0 |
| `kq * KP` where `kq` is provably 0 | still a dynamic shared-memory offset |

Keep Python ints in Python — `range()`, `if` on a constant, a flat buffer — and
let symbolic values appear only as data indices. **Where a refactor must be an
identity at its default setting, make it identity by construction (emit
nothing), never by argument.** Every kernel rewrite here that skipped that step
shipped a wrong number first.

## Commits

Commitizen `<type>(<scope>): <subject>`, scopes `kv` `engine` `ops` `autograd`
`train` `server` `docs` `bench`. Straight to `main`, small self-contained
tranches. The body carries the measurement and the reasoning — a perf commit
without a number in it is not reviewable.
