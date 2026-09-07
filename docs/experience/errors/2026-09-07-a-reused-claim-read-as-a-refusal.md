# A claim that was reused read as a refusal, and the launcher killed a healthy server

**Status:** closed — `scripts/pod_run.sh` tests the exit code as of this entry
**Date:** 2026-09-07
**Card:** H20 card 0, `qwen38-27b` NVFP4, tree `/work/tilerl-s-tierbench` sha `f038c96`

## Context

Tier bench cell 1 (half state budget, 12 sessions, tier off). `pod_run.sh tb1 0 --
bash /work/tierbench_arm.sh off 12 1073741824`.

The launch worked. `pod_run.sh` claimed card 0, the 27B loaded, `/health` answered
in **11 ms**, and the arm script was about to run the bench. Then:

```
pod_run: claimed 0 for tilerl-tb1                     # 13:54:24, pod_run's own block
pod_run: claim pending for 2501011, polling up to 300s for a device fd
pod_run: card_claim FAILED, killing 2501011: tilerl-tb1 already holds 0 for pid
         2501011 (same pid, same cards -- claim reused, not re-taken)
pod_run: exit 4  stat=reaped                          # 14:00:17, 6 minutes later
0, 0 MiB
```

Six minutes of card 0, one 27B load, zero rows.

## Root cause

`pod_run.sh:81-83` states the contract: *each arm re-claims its own python pid as
it starts, never the wrapper's*. The arm script obeyed it — `pod_run_claim $SRV`
on the server's pid.

But `pod_run.sh`'s own block had already resolved the wrapper's claim to that same
descendant python: the claim it printed at 13:54:24 names **pid 2501011**, the
server, not the wrapper 2501008. So the arm's re-claim asked for a pid the name
already held, which is `card_claim.py`'s no-op path (`card_claim.py:1045-1049`):

```python
if holder == old:
    return True, (f"{name} already holds {','.join(cards)} for pid {old} "
                  f"(same pid, same cards -- claim reused, not re-taken)")
```

`ok` is `True` and the process exits **0** (`return 0 if ok else 1`). But the word
`claimed` never appears in that sentence, and `pod_run_claim` tested for the word:

```bash
case "$out" in
  *"claimed"*) echo "pod_run: $out"; return 0;;
```

So a success was read as a refusal, the loop ran its full `DEVICE_WAIT=300`, and
the failure branch killed a server that was healthy and holding a valid claim.

Two correct behaviours combined into a kill: pod_run's helpful descendant
resolution, and the arm's obedience to pod_run's own re-claim rule. Neither is
wrong alone. The substring test is what could not tell them apart.

## Fix

`pod_run_claim` reads the **exit code**, which is what `card_claim.py` actually
publishes as its verdict:

```bash
out=$(python3 .../card_claim.py acquire ... --wait-for-device 1 2>&1) && rc=0 || rc=$?
[ $rc -eq 0 ] && { echo "pod_run: $out"; return 0; }
```

The `ZOMBIE` branch stays: that message comes back on a *refusal*, so it is only
reachable on the non-zero path.

The selftest needed two changes before it could see this, and the first is the
larger one:

**The mock exited 0 on refusals.** `pod_run_selftest.sh`'s fake `card_claim.py`
printed the real refusal *messages* but always returned 0, so under an rc-based
runner every refusal reads as a grant — arm 2, the kill control, would have gone
green against a runner that never kills. The mock now carries the real exit codes,
1 on each refusal path.

**Arm 5, the reuse arm.** `CLAIM_MODE=reuse` replays the exact no-op sentence with
rc 0 and no `claimed`. Negative control run: the new arm against the **old**
`pod_run.sh` fails with `rc 4` — the same exit code the pod produced — and passes
against the fix, with all five arms green.

## Which of the success paths the word match killed

Enumerated over every `return True` in `card_claim.py` rather than reasoned about,
since the question is exactly "what else is there":

| line | outcome | says `claimed`? |
|---|---|---|
| `:1085` | granted | **yes** |
| `:1056-1058` | rebind — new pid is a descendant | falls through to `:1085`, so yes |
| `:1046` | reuse — same pid, same cards | **no** — the bug above |
| `:887` | the asking pid's ancestor already holds the card; **no claim written** | **no** |
| `:639` | `grant-lane` (a different subcommand) | **no** |

So `acquire` has **two** exposed paths, not one. `:887` is the second: a wrapper
whose ancestor holds the card gets `True` with no claim file written and no
`claimed` in the sentence, which the word match would also have turned into a
kill. Raised by tilerl-48 as a question about the rebind branch — rebind is
genuinely safe, because `os.unlink(mine); continue` sends it back through the
normal grant, but the enumeration that answered it found `:887`.

## Rule

**A launcher's success test must read what the tool publishes as its verdict, not
what its message usually says.** Three of `card_claim.py`'s five success returns
lack the word `claimed`, and two of them are reachable from `acquire`; a substring
match on the common one turns the others into kills. When a wrapper and its arms
both claim the same pid by design, the reuse path is the *normal* path, not an
edge case.

**"Which other paths look like this" is a question to enumerate, not to answer
from the paths you can name.** The fix was already correct when the question was
asked, but the answer "one exposed path" was wrong, and only listing every
`return True` showed it.

**A mock that returns 0 for both outcomes disables every rc-based assertion built
on it.** The refusal messages were faithfully copied from the pod and the exit
codes were not, so the mock encoded half the contract. Check that a mock's negative
path is negative in every channel the code under test reads.
