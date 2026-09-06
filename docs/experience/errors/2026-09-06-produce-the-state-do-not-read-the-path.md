# Six rounds of reading the code path instead of producing the state — 2026-09-06

**Date:** 2026-09-06
**Sessions:** tilerl-25, v100-sm70-fp4-55
**Task:** per-rollout logging (#145) and the four fixes that followed it
**What broke:** six consecutive claims about what a program does, each held by two
sessions who had read the code, each wrong until someone ran the program into the
state and looked.

## Context

#145 added `runs/<id>/rollouts.jsonl` so run 3 could pair a completion's length with
its reward. The question that started it was narrow: what does a killed run lose?

Both sessions read `cli.py`. Both found `length_reward_r` assigned after the loop and
the manifest written inside `_finish`. Both concluded a SIGTERM loses the metric and
keeps the rows. That was wrong, and the five rounds after it were wrong in the same
way — not from bad reasoning about the code, but from reasoning about code at all when
the question was about the filesystem after a signal.

## Root cause

Reading a code path tells you what it does **when reached**. Every question here was
about what is true **when it is not** — after a signal, on the other algorithm, with a
gate list nobody appended to, with a prompt that supplied half a delimiter. No amount
of reading answers that, because the answer is in the state, not the statements.

Six rounds, each fix's own output being the next thing worth probing:

| # | the claim, from reading | what producing the state showed |
|---|---|---|
| 1 | a killed run loses `length_reward_r` | **the manifest did not exist at all** — `tilerl ledger` printed `[]`, `list_runs` globs `*/manifest.json` |
| 2 | the recovered manifest is fine | it prints `skip`; no gate was ever evaluated |
| 3 | the pre-loop write fixes it | it sat inside `if args.rl:` — an interrupted **opd** run left no run *directory* |
| 4 | `pass` is unreachable | reachable, and **created by the round-3 fix**: opd appends no drift gate, so `gates == []` and `gates_pass([])` is `all([])` = `True` |
| 5 | the test protects `format_run` | `test_manifest_round_trip_and_lineage` asserted `pass` on an unfinished manifest — green, encoding the defect |
| 6 | `strip_think`'s self-check covers it | it asserted `<think>…</think>`; the live path emits only `</think>`, so the regex never matched |

Rounds 1–5 landed as #145, #146, #148 ([an interrupted run reported
pass](../wins/2026-09-06-an-interrupted-run-reported-pass.md) is round 5's own entry).
Round 6 is [the reply carried the reasoning and a bare
closer](2026-09-06-the-reply-carried-the-reasoning-and-a-bare-closer.md), found from a
user report rather than a probe, and it is the same defect with the operands swapped.

### Two of mine

**I reported a verdict from a dict I built by hand.** Round 4's `pass` came first from
constructing what I believed `write_manifest` puts on disk and calling `format_run` on
it. The peer corrected it — unreachable on grpo, because `manifest["gates"].append(drift)`
precedes the write — and the correction was right. `pass` turned out to be reachable
anyway, on the opd path, which I then found by killing a real opd run. **The conclusion
survived and the method did not**: I had the right answer for a reason that was wrong,
which is indistinguishable from being right until someone checks.

**I quoted two line numbers off a stale copy.** `format_run`'s consumers are `cli.py:822`
and `:970`; I said `:258` and `:405`, read from a `/tmp` dump taken before two merges
shifted the file. Same failure one level down: an artifact standing in for the tree.

### Two of the peer's

**"The `pass` branch is unreachable"** — right for grpo, wrong once its own hoist created
a path with no `append(drift)`. A correct statement about the code as it was, made about
the code as it had just become. Its own correction to this line is the part worth
keeping: the claim was made **by reading**, in the round whose whole subject was that
reading a path is a hypothesis about it.

**Credit for `gates_skip_after` given to the session that relayed the review**, not the
one that wrote the fix. The relay-channel error at conversation scale — and the grant
commit read "via tilerl-27", which is the exact shape `AGENTS.md`'s addressing rule
already warns about. A rule read and still walked into, not a gap in the rules.

### The instrument lied once

A probe waiting on opd step lines reported "0 after 180 s", which reads as a stalled
loop. The opd step line calls `log` without `flush=True` while the grpo one passes it,
so a pipe buffered every line. Under `python -u` the loop was running normally. **A
stdout-driven probe cannot distinguish a quiet loop from a stopped one when what it
reads is buffered** — and a plausible number is worse than an error, because it gets
believed.

A second instrument the same night reported a partition of its data instead of the data:
a `/health` probe printed a hand-listed tuple of keys that omitted `pool_used_blocks`,
which had been in every response from round one (`grep -c pool_used` on that log: 0).
Four rounds went to a question the first response had already answered. Same family as
the hand-built dict above — the fixture was written rather than driven, and the output
looked complete.

### And the recovery raced

#145 was squash-merged at head `8980eb8` while two commits were still being pushed to
its branch. `59848e3` (the opd hoist) and `f9410e1` never reached main, both at
`total_count 0`. Nothing showed it: `gh pr view` said MERGED and the branch ref still
pointed at the newer sha. `merge-base --is-ancestor` cannot help — a squash makes every
branch commit a non-ancestor, so it reads identically for "squashed in" and "missed".
Only `git show origin/main:<file>` showed the hoist still inside `if args.rl:`. Then both
of us independently opened byte-identical recovery PRs (#146, #147) an hour apart.

**`total_count 0` has three meanings and the count cannot separate them.** A stranded
commit on a merged branch; a fresh branch before its PR opens; and — measured on #149 —
a PR whose `mergeStateStatus` is DIRTY, because GitHub queues no `pull_request` event
while it conflicts. That last one looks exactly like the first two: pushed, saw `0`,
neither fresh nor stranded, just a CHANGELOG line that had collided with #148 after it
merged. The discriminator is `gh pr view --json mergeable`, not the run count. Rebasing
queued the event immediately.

## Fix

Landed as #145 `2fc2697`, #146 `38fd5fc`, #148 `9fcafb9`, #151 `eccac47`. Current state
on main: `format_run` checks `finished` first and returns `killed`; the manifest write
is above `evals("before")` for both algos; `strip_think` takes `opened=`.

One case is deliberately open — a *finished* manifest with `gates: []` still reads
`pass`, and `cmd_merge` always produces exactly that shape because it calls
`write_manifest` directly and never `_finish`. Every `tilerl merge` row is a verdict
over zero checks. Ruled: it will render `none`, so `skip` keeps meaning "a gate existed
and was suppressed".

## Rule

**Produce the state; do not derive it.** For any question of the form "what is on disk
/ what does it print / what survives when X does not happen", run the program into X and
look. Reading tells you what the path does when taken. Two sessions reading the same
path is one reading — the redundancy goes to zero while feeling unchanged
([two-agents-one-broken-instrument](2026-09-05-two-agents-one-broken-instrument.md), the
same finding through a different instrument).

**Probe the fix's own output.** Every round here was found by probing the previous
round's fix, and round 4's defect was *created* by round 3's. A fix changes the
reachable state space; the new space has not been looked at.

**A green test that asserts the wrong answer is worse than no test.** It inverts the
burden of proof: the correct fix must now argue for changing a passing assertion, which
reads as weakening a test. Rounds 5 and 6 are the two forms — the wrong *answer*
(`pass` on an unfinished manifest) and the wrong *input* (`<think>…</think>` where the
live path emits only `</think>`). Before trusting a test to protect a path, check
whether its fixture is a string the system actually produces. That also says where the
next one is: **any assertion whose fixture was hand-written rather than driven through
the code under test.**

**Verify a merged commit by content, and check CI the moment you push.** `git show
origin/main:<file> | grep <the line>` is the late detector. The early one is
`actions/runs?head_sha=<sha>`: a commit pushed to a merged PR's branch gets no runs at
all. But `total_count 0` means three different things — stranded, fresh-before-PR, or
conflicting — so the count is never the answer on its own. Pair it with `gh pr view
--json mergeable`: a PR that exists, is not DIRTY, and still reads `0` after a push is
the stranded case. The three detectors only work as a set; each one alone cannot say
when it applies.

**Say "no more pushes" when handing a branch back**, and re-read the head immediately
before merging. Both halves of this one failed at once: the merger read the head minutes
early, and the pusher said "green is yours" and then pushed twice more.

**Announce a recovery before opening it.** Two sessions watching the same queue produce
duplicate PRs by default, not by accident.

**Errors that favour you get less scrutiny.** Two numbers withdrawn the same night — a
block table that multiplied per-entry blocks by entry count when blocks are refcounted
across nested prefixes, and a crossover extrapolated from an average that rises
monotonically — both ran toward the cheaper, more favourable conclusion. Nothing in this
entry was caught because it looked wrong; the ones that survived longest looked right.
