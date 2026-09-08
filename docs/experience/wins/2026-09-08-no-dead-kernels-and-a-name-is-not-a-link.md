# The kernel layer has no dead code, and a name is not a link — 2026-09-08

> Status: audit of `packages/tilerl-kernels/` (9 files, 7639 lines) and `tests/`
> (52 files, 16931 lines). **Zero dead kernels, verified three ways. 16 assertions
> whose tolerance guards a measured-zero effect, of which 7 are free cleanups.**
> PR #284 tightens the one with a structural proof. Dev-only, exempt from the
> bench gate. No CHANGELOG line here: the audit's verdict spans three lanes
> (scripts/, kernels+tests, src/tilerl/) and is one line about the tree, written
> once when all three close.

## The claims, separable from the story

1. **The kernel layer has no dead code.** 111 module-level functions, 59 registry
   names, 44 backend dispatch keys, 49 `reference.py` functions — 0 unreachable in
   any of the four enumerations.
2. **A name is not a link.** Four instruments failed permissively in one audit
   because the real link is unnamed: a factory `return`, a CI glob, a filesystem
   round-trip, a name passed as an argument. Table below.
3. **The dangerous band is the middle of the ratio.** 43/44 and 113/200 are
   rejected on sight; 4/44 would have been believed and would have deleted live
   dispatch. Validate a sweep against a known-reached case first.
4. **Measuring at the boundary beats tracing structure — only while the boundary
   is live.** Where the link is historical, hand-reading is the method and its
   cost is real.
5. **Zero measured deviation is a symptom with three causes** (forced-const,
   forced-same, incidental), and they take three different actions.
6. **Tighten-to-exact buys the dtype's precision, not arbitrary precision.** State
   it relatively; the absolute restatement is false over a wide magnitude range.
7. **Mutate after a tightening, not only before**, and record what it still cannot
   catch.
8. **A comment naming a failure mode is as likely to be a defence as a
   confession.**

## Context

Full-tree audit to consolidate and delete dead code. Two halves with different
methods: kernels are about reachability, tests are about whether a green check
certifies anything.

## Half A: nothing is dead

Three independent enumerations, each against a different entry-point set:

| enumeration | declared | reached | unreachable |
|---|---|---|---|
| module-level public functions | 111 | 111 | **0** |
| registry factory names | 59 | 59 | **0** |
| backend dispatch keys | 44 | 44 | **0** |
| `reference.py` public functions | 49 | 49 | **0** |

All four arch cells (`cpu`, `metal`, `sm90`, `sm70` × `bf16`/`fp4`) are
registered and live. The ROCm alias is gone from code — the six surviving
mentions are historical prose in `docs/experience/` and should stay.

So the answer to "what can we delete from the kernel layer" is nothing, and that
is a result about the project rather than a failed search.

## Half B: the workflow that beat reading

Reading found one suspect. **Measuring found sixteen — and the measuring was the
cheaper half.** One 239 s run, suite still green, covering all 52 files; the
reading covered a handful of the widest-tolerance sites and produced one. The
instrumented run is not merely better evidence, it is the shorter path, and it
scales to files nobody opened.

Monkeypatch `torch.allclose` in `tests/conftest.py`, record max|Δ| and max
relative Δ against the declared band at every call, run the suite once:

```
239 s, 481 passed, 15 skipped, 6 xfailed   (green -- the probe changes nothing)
32 allclose sites measured
16 positive assertions with an effect of EXACTLY zero
```

Then classify, then mutate only the survivors. Cheap enough to re-run on any
suite. **Do not start by reading tests.**

### Zero is a symptom with three causes

The partition is what decides the action, and it is three-way, not two:

| class | sites | action |
|---|---|---|
| **forced-const** — one side is a literal the test wrote | 5 | tighten, free |
| **forced-same** — both sides reach one producer | 2 | tighten, free |
| **incidental** — different computations agreeing bit-for-bit today | 3 | leave; tightening is a *new* determinism claim that a compiler bump can break |
| **unmeasured arm** — band may guard a target that skips here | 2 | leave, record why |
| **not vacuous** — band sits over a real comparison | 4 | nothing |

`test_merge.py:97` is forced-same: `merge_checkpoints` reaches
`iso_merge_weight` at `merge.py:144`, `iso_merge` at `:79`, so the merge math
cancels and 0.000e+00 over 17 tensors is *forced*, not observed.

### Tighten to exact buys the dtype's precision, not arbitrary precision

Measured on PR #284's own assertion. Scaling the shard write:

```
x1.0000001 … x1.001   survive     x1.01, x1.05   caught
```

`torch.equal` is exact *in bf16*, and bf16's mantissa gives a ~3.9e-3 relative
step. **The detection floor is the dtype, not the tolerance.**

Do not restate that as an absolute equivalence. These 17 tensors span |x| from
1.423e-06 to 4.562 — six orders — so "`torch.equal` ≈ `atol=3e-3`" is wrong by
~3 orders at the small end, where `atol=3e-3` passes nearly every distinct bf16
value. The relative claim holds unconditionally; the absolute one does not
survive the range.

### Mutate after the change, not only before

Tightening `:97` does **not** make it catch merge-math regressions, and never
could: `ridge` 1e-3→1e-1 and dropping the tangent projection both still pass,
because both sides move together. The structural fact that justifies the
tightening is the same one that limits it. What it does catch, verified: a lossy
shard write (`.half()` inserted) turns it red where the old band did not.

Skipping the after-mutation would have shipped a tightening implying a guarantee
it does not provide — a vacuous assertion at a new tolerance.

## The ceiling: a name is not a link

Four false results in one audit, all the same shape — **I searched for names and
the codebase addresses things by value.**

| the link | what a name search saw | would have deleted |
|---|---|---|
| kernel returned by its factory's `return` | 3 "dead" kernels | 3 live kernels |
| CI glob `tests/*_world[0-9].py` | "invoker: NONE" ×9 | 9 live CI gates |
| two sides meeting through `save_file`→`load_hf` | detector cannot see its own motivating case | — |
| `self._kernel(name, …)` with the name as an argument | **43 of 44 keys unreachable** | live dispatch |

Every one failed in the permissive direction. 43/44 only failed safe because the
ratio is absurd on sight — **at 4 of 44 it would have read as a finding.**

`tilerl-48` hit the same ceiling from the opposite side in the same audit: every
script docstring says `Run: scripts/<self>.py`, a self-loop that made 113 of 200
files look imported. **An edge that almost always holds and an edge that almost
never holds fail identically — neither discriminates.** Both of us were saved by
implausibility, not by method.

They also found the parameter-level version: `--out` appears in 24 files with 11
different meanings. Same name, different semantics.

### Claim: the dangerous band is the middle of the ratio

An instrument's output is credible only where its base rate is informative.
43 of 44 "unreachable" and 113 of 200 "imported" are both rejected on sight.
The band that gets believed is the plausible one: **4 of 44 unreachable would
have been filed as a finding and deleted live dispatch.** So a ratio near the
extremes is evidence about the instrument; a ratio in the middle is evidence
about nothing until the instrument is validated against a known-live case.

Before trusting any reachability sweep: run it against something you already
know is reached, and check it says so.

## Claim: measuring at the boundary works only while the boundary is live

The escape from the unnamed-link ceiling is to stop tracing structure and measure
the quantity at the call. That is what turned one finding into sixteen here, and
it is **not general.**

The condition: the link must have a runtime observable *at measurement time*.

- **Live boundary — measure it.** Every `torch.allclose` in this suite executes on
  every run, so one patched `conftest.py` observes all 32 sites for 239 s.
- **Historical link — hand-read it, and the cost is real.** `tilerl-48`'s
  probe→entry link has no runtime observable: the entry is the artifact of a run
  whose process is long gone, so re-running the probe yields a *new* number rather
  than the one recorded. The quantity that would prove the link exists only in the
  past. They read 34 files by hand; there was no cheap version.

Stated so the next reader does not reach for `conftest.py` and find nothing to
patch.

## A comment naming a failure mode is as likely to be a defence as a confession

`test_e2e.py:632` carries the message *"…two identical fresh computations and this
test is inert"* and was ranked prime suspect on that phrase. It is the message of
a `prefix_hits >= 1` guard **against** that failure, and it passed — so a hit
occurred, the two paths differed, and zero deviation means the restored GDN state
is bit-identical to the recomputed one. The strongest result that test can
produce. Reading three lines up is how you tell.

Related: an instrument that scores every site uniformly misreads the negations.
Three of the 32 sites are `assert not torch.allclose(...)`, where a large
deviation is the *pass* condition; my first ranking put them at the bottom
looking like the worst offenders.

## Rule

- Enumerate the entry-point set, **and check whether its members are addressed by
  name at all.** Where they are not — factory return, glob, dict value,
  name-as-argument, a path through the filesystem — absence of a name proves
  nothing.
- Judge a tolerance by measuring the effect it guards, in one instrumented run,
  before reading any test.
- Zero measured effect is a symptom. Forced-const and forced-same are free to
  tighten; incidental is a new claim; an unmeasured arm is a reason to leave it.
- Mutate **after** a tightening, not only before, and state in the commit body
  what it still cannot catch.
- Tighten-to-exact buys the dtype's precision. Say so relatively, never as an
  absolute band.
- Before pushing, `git diff origin/main | grep -c '^-[^-]'`. Costs a second and
  catches both a bad rebase and a whole-file overwrite. Better still: never write
  a whole file that several sessions append to.
