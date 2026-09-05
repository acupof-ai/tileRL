# I read a two-variable condition as a structural dead end — sm70, 2026-09-05

> Status: the verdict was withdrawn; the tier ships opt-in

## Context

Building a pinned-host tier for GDN snapshots. Under `state_bytes` pressure the store
demotes the LRU snapshot instead of evicting the entry, and a later hit promotes it
back. The question was whether a demoted snapshot is ever asked for again.

Three measurements, in the order I took them:

| probe | promotions | demotions |
|---|---:|---:|
| one conversation, V100 | 0 | 43 |
| two interleaved, V100 | 0 | 16 |
| CPU sweep at 2 conversations | 6 only at `state_bytes`=2 | — |

I concluded the tier was structurally dead: "demotion picks the LRU end, a lookup
always wants the MRU end, they never meet at any budget above 3 snapshots." I wrote
that into the design page as a rejection and reported it as a mechanism rather than a
tuning result.

## Root Cause

**The claim is true within one conversation and false across conversations.** Across
sessions the LRU end *is* some other session's newest entry, so a lookup reaches it as
soon as there are more live conversations than the budget holds. A peer said exactly
this; the sweep I had already run said it too, and I had read it the other way.

Rotating N conversations against a fixed 9-snapshot budget:

| conversations | promotions | demotions |
|---:|---:|---:|
| 2 | 0 | 0 |
| 4 | 0 | 3 |
| **9** | **17** | 35 |
| 12 | 24 | 51 |
| 20 | 40 | 91 |

At 12 conversations the tier is the difference between no reuse and complete reuse:
**0 hits / 63 evictions without it, 24 hits / 0 evictions with it.**

So the condition is `sessions > budget`, a relation between two variables. My earlier
sweep held sessions at 2 and varied the budget; it found "only budget 2 works," which
is the same rule with the other operand pinned. Reading a two-variable condition off a
one-variable sweep is what produced a structural claim from a conditional one.

Both of my probes had the same blind spot because both came from the same workload I
was already running — a single chat script, then the same script twice.

## A second instrument was wrong in the direction of my conclusion

An idle-card probe put a 144 MiB pinned D2H copy at **11.55 ms**; in situ it is
**161.9 ms**, 14.0x, because a demotion happens mid-prefill where the copy contends for
the link and forces a sync. Even that only attributes **7.0 s of a 68.7 s** regression
(10%) — the rest is contention in the prefill it interrupts, which a per-call timer
cannot see. I had quoted the 11.55 ms to argue the cost was negligible, and later
quoted the 161.9 ms to argue the tier was expensive. Neither number was load-bearing;
the promotion count was.

## Fix

`dram_bytes` defaults to **0** — correct for the single-session endpoint this pod runs,
wrong above 9 concurrent sessions, and the flag is what distinguishes them. The design
page now states the condition instead of a verdict. Still unmeasured: the V100 wall
clock at 12 sessions; the numbers above are hit counts, not time.

RL rollouts are not a second case — within a group the shared prompt is the MRU entry
and the store is cleared between steps, so nothing ages out and returns.

## Rule

Before calling a negative result structural, name every variable in the condition and
vary each one. A sweep over one operand of a two-operand relation returns a threshold
on that operand and reads exactly like a law. Here `sessions` was fixed at 2 in every
probe I ran, because every probe came from the workload already in front of me — so the
sweep that looked like a control was a control over the wrong axis.

And a rejection deserves the same scrutiny as a claim. Two instruments erred toward my
conclusion — a budget that suppressed promotions and an idle card that made the cost
look free — and neither produced a surprise, which is why neither got checked.
