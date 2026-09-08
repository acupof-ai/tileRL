# P1's exit gates cannot see the target P1 was set — 2026-09-08

> Status: **open**, and the fix is not mine to land — `tilerl-0a` owns `cli.py:833`. This
> entry is the design input: which gates have a zero threshold, what floor the test they
> currently apply has, and which test the run's own data already supports.
> Instrument: `scripts/gate_noise_floor.py`.

## Context

The P1 roadmap exit criterion is *"GSM8K held-out 500 q, after − before ≥ +5 pt"*. The gate
in the tree (`cli.py:831-837`) is five bare inequalities on pairs of scalars:

| gate | threshold |
|---|---|
| `reward_rises` | `reward_last > reward_first` |
| `gsm8k_improves` | `gsm8k_after > gsm8k_before` |
| `ce_falls` | `ce_last < ce_first` |
| `mmlu_holds` | `mmlu_after >= mmlu_before − 0.03` |
| `groups_untied` | `tied_group_fraction < 0.5` |

**Three of the five have a zero threshold.** A zero threshold is a coin flip: simulated at
n=500, p=0.40, `after > before` passes **48.7%** of the time when nothing has changed. The
1.3 pt under 50% is the discrete binomial's tie mass, which a strict `>` fails — that is the
only thing holding it under a coin flip, and it shrinks as n grows. This number does not
depend on which statistical test is used, so it is the one to quote.

## The finding: the roadmap's own +5 pt is below the floor of the test the gate applies

Two independent binomials at n=500, p=0.40 give SE(diff) = **3.10 pt**, so 80% power at 5%
significance needs **7.70 pt**:

| gate | threshold | SE | min detectable | verdict |
|---|---|---:|---:|---|
| `gsm8k_improves` | `after > before` | 3.10 pt | 7.70 pt | below the floor |
| **gsm8k roadmap** | **≥ +5 pt** | 3.10 pt | **7.70 pt** | **below the floor** |
| `mmlu_holds` | within −3 pt | 3.16 pt | 7.86 pt | below the floor |

So tightening `after > before` to `≥ +5 pt` would replace a gate that passes on noise with a
gate that **rejects real +5 pt gains most of the time** — a different failure, not a fix. The
threshold is not the whole lever.

## The run already writes the data for a test that can see it

`_write_eval_rows` (`cli.py:358`) states the situation in its own docstring:

> One JSON row per problem, so two arms over the same set can be compared **paired**. …
> **P1 fell back to the unpaired interval because only totals were kept.**

Checked, and all three preconditions hold now:

1. **Same slice both arms** — `eval_rows = _jsonl(args.eval_gsm8k)[:args.eval_n]` is computed
   once at `cli.py:479` and used by both `evals("before")` and `evals("after")`, at
   temperature 0.
2. **Per-problem rows land on disk** — `_write_eval_rows` writes `eval-{tag}.jsonl`, and
   `eval.py:104` carries `correct` per question.
3. So **McNemar applies**: the paired SE is `sqrt(b + c) / n` over the discordant counts.

The floor then scales with the **flip rate**, not the accuracy:

| discordant `(b+c)/n` | SE | min detectable | +5 pt |
|---:|---:|---:|---|
| 30% | 2.45 pt | 6.09 pt | below the floor |
| 20% | 2.00 pt | 4.97 pt | resolvable |
| 10% | 1.41 pt | 3.52 pt | resolvable |
| 5% | 1.00 pt | 2.49 pt | resolvable |

**+5 pt becomes resolvable once fewer than ~13% of questions flip** — which is the regime a
LoRA RL step actually lives in. The unpaired formula cannot express this at all: it has no
term for how many questions moved.

**`(b + c)` is not knowable in advance.** It is what a run has to report, and today's
manifest does not — only `gsm8k_{tag}` (a count) and `gsm8k_{tag}_total`. The rows are on
disk; nothing reads them back into the gate.

**One caveat on the paired floor**, 0a's: the before-arm can be served from
`runs/eval-cache/{key}.json` (`cli.py:562-570`) instead of being evaluated, and that path
writes the cached rows. Still paired, but the pairing is only as good as the cache key — so a
paired SE from a cache-hit run is partly a claim about the key.

## The units make the obvious fix a no-op

`tilerl-0a`'s finding, verified: **`gsm8k_{tag}` is an int count** (`cli.py:589`,
`metrics[f"gsm8k_{tag}"] = c`) while **`mmlu_{tag}` is a fraction** (`:576`, `c / n`). Pinned
by `tests/test_ledger.py:71` and `:102-103`, which assert `isinstance(..., int)`.

Three consequences:

- `gsm8k_improves` is `after > before` on counts, so its real threshold is **+1 question =
  +0.2 pt**, not zero.
- **`gsm8k_after - gsm8k_before >= 0.05` on integers is `>= 1`, which is `> 0` — the same
  gate.** Writing the roadmap's `+5 pt` that way changes nothing while looking like a fix.
  0a nearly shipped it. The correct encoding is `>= 0.05 * total`, i.e. **25 questions**.
- `mmlu_holds` is genuinely mis-numbered rather than mis-united: the code uses
  `mmlu_before - 0.03` where the roadmap says **−2 pt**. A real 1-point loosening.

So the gate block has three separate problems that all look like one: a threshold below the
noise floor, a unit mismatch that silently absorbs a fractional threshold, and one constant
that disagrees with the roadmap.

## Rule

**A zero-threshold gate is a coin flip, and that is a property of the threshold, not of the
noise.** Three of five gates here are zero-threshold. No amount of sample size fixes them;
only a threshold above the floor does.

**Price a threshold against its test's resolution before tightening it.** The obvious fix —
raise `after > before` to the roadmap's `≥ +5 pt` — swaps a false-pass gate for a
false-reject gate, because the test in place cannot resolve 5 pt at n=500. Both settings are
wrong for the same reason: the test, not the number.

**When code says why it settled, read it before rebuilding the analysis.** `cli.py:358` names
the paired comparison, the reason P1 did not use it, and the fact that the reason has since
been removed. My first pass computed unpaired intervals and would have handed over a floor
roughly 2x too high — conservative in direction, but wrong on exactly the quantity the gate
needs. Credit to `tilerl-27` for pointing at the docstring.

**An instrument's assert should encode the finding, not the expected pass.** The first
version asserted the +5 pt target clears its floor. It failed, and the failure *was* the
result. The assert now states the finding in both directions: unpaired cannot resolve 5 pt,
paired at a 10% flip rate can — plus a control that the paired SE tracks the flip rate rather
than the accuracy, which is the one way this instrument could be silently wrong.

**A threshold's units are part of the threshold.** Three sessions looked at this gate block
tonight and the unit mismatch was the finding none of the statistics would have caught: on an
int metric, a fractional threshold is absorbed silently and the gate keeps its old behaviour
while its source reads as fixed. 0a caught it by checking what the metric *is* before
changing what it is compared against.

## Hand-off

Owner of the fix is `tilerl-0a` (`cli.py:833`). This entry supplies:

| gate | zero-threshold? | today's real threshold | what it should be |
|---|---|---|---|
| `reward_rises` | yes | any increase | — (not a P1 exit metric) |
| `gsm8k_improves` | effectively | +1 question (+0.2 pt) | `>= 0.05 * total` = 25 questions, McNemar |
| `ce_falls` | yes | any decrease | — (not a P1 exit metric) |
| `mmlu_holds` | no | `−3 pt` | `−2 pt` per the roadmap |
| `groups_untied` | no | `< 0.5` | unexamined here |

and the resolution numbers above. The one number to quote when arguing the gate is unfit:
**a near-zero threshold passes 48.7% of the time when nothing changed.**
