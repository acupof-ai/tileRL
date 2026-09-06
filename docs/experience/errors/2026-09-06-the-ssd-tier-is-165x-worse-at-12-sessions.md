# The SSD tier is 1.65x worse per turn at 12 sessions, and it serves nothing to pay for it

**Status:** closed
**Date:** 2026-09-06
**Commit:** a5b63cb
**Card:** H20 card 6
**Verdict:** REJECT on the serve path at this session count. The tier fails #152's criterion
at the only session count where the criterion can be evaluated.

## Context

#152's rule: the tier ships on the serve path only if wall clock per turn is **not worse**
at the session count where HBM overflows. The drain side, the arrival side and the cap were
all measured earlier today
([max_pending](2026-09-06-the-max-pending-cap-is-not-the-queue-that-binds.md)); the wall
clock the verdict actually turns on was not.

## Measured

Six cells, two arms, one server process per arm, `--max-ctx 49152 --slots 3`, 12
conversations x 3 growing turns. A warm-up arm ran first so no measured cell pays a first
compile: **`compiles` is 0 in all 132 rows**, asserted rather than hoped for.

| sessions | turns/arm | off mean s | on mean s | on/off | pool peak | prefix evictions | ssd offered | ssd hits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 1.447 | 1.582 | 1.093 | 28.4% | 0 | 12 | **0** |
| 8 | 24 | 1.631 | 1.682 | 1.031 | 99.8% | 144 | 60 | **0** |
| 12 | 36 | **1.971** | **3.256** | **1.652** | 99.5% | 491 | 132 | **0** |

Only the 12-session row is a difference. Welch on 36 turns per arm: diff **1.285 s**,
se 0.331, **t = 3.88**, 95% CI on the ratio **[1.32, 1.98]x**. Medians agree and are
harsher: 1.562 → 3.078 = **1.971x**. The 2- and 8-session rows are **not** differences:
t = 0.53 and t = 0.22, CIs spanning zero ([-0.36, +0.63] s and [-0.40, +0.50] s). Reporting
those as small costs would be reading noise as signal.

## Both thresholds, because they are not the same threshold

I said before running this that "the session count where HBM overflows" was a quantity to
measure, not 12 by assumption. It resolved, and the two ceilings turn out to be separated:

- **The block pool saturates between 2 and 8 sessions**: 28.4% at 2, **99.8% at 8**. So HBM
  overflow arrives at 8, not 12, and 8 is where the tier is supposed to start earning.
- **The SSD byte budget starts thrashing between 8 and 12**: `ssd_evictions` 0 → 10 → 79.

At 8 sessions — the actual HBM-overflow point — the tier is **1.031x, indistinguishable
from free**. The 1.65x penalty appears only at 12, where the tier's own byte budget is
evicting 79 entries. So the cost is not the cost of spilling under HBM pressure; it is the
cost of spilling into a full tier that then throws the entries away.

## Why it can only cost: the store publishes and never serves

**`ssd_hits` is 0 in every one of the 132 rows, across 132 offers.** This is not a discovery
and I nearly wrote it up as one — `AGENTS.md:162` already states it: "today's store
publishes and never serves," which is why the RL path runs `prefix_store=NoPrefixStore()`.
Checked before claiming a mechanism.

That makes the arithmetic one-sided. A tier with a hit rate of 0 has no credit side at all,
so every measurement of it is a measurement of pure cost, and the only question the wall
clock can answer is how much. At 8 sessions the answer is "nothing measurable"; at 12 it is
1.65x.

**So the reject is narrow and the reason matters for what comes next.** This does not say
an SSD tier is a bad idea. It says *this* tier, whose load path is unreachable, cannot repay
its write cost, and the write cost becomes visible exactly when its byte budget starts
churning. The block-granular store that `AGENTS.md` names as the upgrade would change the
credit side and the verdict has to be re-run against it — it does not inherit this one.

## What I got wrong in the instrument

`prefix_entries` is **not a key the engine emits**. I read it in every row and
`.get("prefix_entries", 0)` returned a plausible 0, which I briefly read as "the store holds
nothing" — a finding about the system inferred from a typo in my own probe. The real keys
are `prefix_hits`, `prefix_misses`, `prefix_published`, `prefix_evictions`,
`prefix_state_bytes`, `prefix_state_bytes_budget`, `prefix_demoted`. `ssd_hits` **is** real
(`kv_cache.py:1069`) and its 0 is the system.

The rows are still valid: `prefix_evictions` and `pool_used_blocks` are real keys and carry
the threshold argument. But a `.get(k, 0)` on a key that does not exist is indistinguishable
from a measured zero, and this is the second time today a default has stood in for data.
The fix is to assert the key set once at startup rather than defaulting per row.

## Rule

**A ratio needs its spread before it is a finding.** Three ratios came out of this run and
only one survived: 1.093 and 1.031 are noise (t = 0.53, 0.22) and would have read as "a
small penalty that grows with load" if quoted as point estimates. The story they suggested
was wrong in shape, not just in magnitude — the penalty is not gradual, it appears when a
second mechanism starts firing.

**Measure where the criterion says, not where the previous run happened to sit.** The
criterion pointed at HBM overflow; HBM overflow is at 8; 12 was inherited from the arrival
probe's configuration. Had I run only 12, the verdict would have been right and its reason
wrong — attributed to HBM pressure instead of to the tier's own eviction churn.

## Results

| date | commit | card | arm | metric | value |
|---|---|---|---|---|---|
| 2026-09-06 | a5b63cb | H20 6 | 12 sessions | off / on mean per turn | **1.971 / 3.256 s** |
| 2026-09-06 | a5b63cb | H20 6 | 12 sessions | on/off, 95% CI | **1.652x [1.32, 1.98]** |
| 2026-09-06 | a5b63cb | H20 6 | 12 sessions | on/off, medians | 1.562 / 3.078 = **1.971x** |
| 2026-09-06 | a5b63cb | H20 6 | 8 sessions | on/off | 1.031x, **t = 0.22, not a difference** |
| 2026-09-06 | a5b63cb | H20 6 | 2 sessions | on/off | 1.093x, **t = 0.53, not a difference** |
| 2026-09-06 | a5b63cb | H20 6 | all arms | pool peak at 2 / 8 / 12 | 28.4% / **99.8%** / 99.5% |
| 2026-09-06 | a5b63cb | H20 6 | on arm | ssd_evictions at 2 / 8 / 12 | 0 / 10 / **79** |
| 2026-09-06 | a5b63cb | H20 6 | on arm | ssd_hits, 132 offers | **0** |
| 2026-09-06 | a5b63cb | H20 6 | all cells | compiles | **0** (asserted) |

## Limitations

- **Serial**: `--max-batch 1`, one request in flight, so a "session" is a conversation the
  server round-robins, not concurrent load. Concurrency changes both arms.
- **Three turns**, growing to ~8.4k prompt tokens. A longer conversation spills more and
  the 12-session ratio is specific to this shape.
- **One `--max-ctx`/`--slots`.** The pool ceiling is what puts HBM overflow at 8; a
  different pool moves that threshold and therefore where the criterion is evaluated.
- The 8-session cell being non-significant is a **failure to detect a difference**, not
  evidence of none: the CI admits up to +0.50 s, which is 31% of the off mean.
