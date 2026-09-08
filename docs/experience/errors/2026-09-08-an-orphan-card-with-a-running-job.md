# A card read as an orphan while its job was running, and smi's pid could not claim it — 2026-09-08

**Status:** fixed at the call site. Card 2 held 66301 MiB with no claim for several minutes
while my own job was in state `Rl` on it. Two separate defects: one made the claim lapse, the
other makes the tool the alert recommends unable to resolve it.

## Context

`tilerl-48` reported `ORPHAN card 2 holds 66301 MiB with no claim`. An unclaimed card holding
memory is what gets a container restarted under someone else's run, so it reads as a leaked
process.

It was not. `ps` showed pid 3304585 — `prof_grpo_step.py --group 16` — in state `Rl`, 8:26
elapsed, on warm step 4 of 5.

## Defect 1: a multi-arm wrapper must re-claim per arm, and mine did not

I launched both throughput arms as one job:

```sh
scripts/pod_run.sh tput 2 -- bash -c 'for g in 8 16; do python3 -u scripts/prof_grpo_step.py --group $g ...; done'
```

`pod_run.sh` claims the card for the python it resolves at launch. That is arm 1's python;
when arm 1 exited the claim was released, and nothing claimed for arm 2.

**This is documented and there is already a correct implementation in the tree.**
`pod_run.sh` exports `pod_run_claim` precisely so a multi-arm wrapper can call it per arm, and
its own comment says a four-arm run "means three windows where the wrapper's claim reads stale
and the card reads orphan". `scripts/tp_step_arms.py`'s `claim_card(proc.pid)` does exactly
this, with a docstring explaining why.

So the mechanism, the warning and a working example all existed. What I wrote instead was a
bare `bash -c` loop, which has no place to hang a per-arm claim. **A launcher that exports a
helper for a shape is telling you that shape needs the helper; using a shell loop skips the
extension point without appearing to.**

## Defect 2: the pid the alert tells you to use cannot claim the card

The orphan alert says: *find it with `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory`.*
Run inside the container, that returns:

```
2882914, 32776 MiB
3543694, 66294 MiB
3579292, 74542 MiB
3560146, 29650 MiB
```

**None of those pids exists in the container.** They are host-namespace pids, and the claim
ledger takes a pid it can check via `/proc`. So the recommended command identifies *that* a
card is busy and by how much, and gives an identifier the next step cannot use. Worse, the
byte figure invites the shortcut: 66294 MiB is close enough to 66301 that it is tempting to
match on it and call the job identified.

**And the tool that prints that advice already knows better.** `card_claim.py:201`, under
"Why fds and not nvidia-smi", records the measurement: *"`nvidia-smi --query-compute-apps`
inside the pod's container reports HOST-namespace pids — 3796003/3796004 while the container's
own ranks were 842276/842277 (measured 2026-09-04)"*, and concludes that matching a claim
against that list "would refuse every correct claim on the pod". The orphan message at
`:1306` recommends the command that docstring exists to reject. The fact was known, measured
and written down four days earlier, in the same file — **a docstring 1100 lines from the
message it contradicts is not a guard**.

What resolves it is the container's own `/proc`:

```sh
ps -eo pid=,stat=,command= | grep prof_grpo_step      # the candidate
tr '\0' '\n' < /proc/3304585/environ | grep CUDA_VISIBLE_DEVICES   # CUDA_VISIBLE_DEVICES=2
cat /proc/3304585/cmdline | tr '\0' ' '                # which arm, which flags
```

`CUDA_VISIBLE_DEVICES` is positive proof of *which card* a live process holds. The memory
figure is not — two jobs of similar footprint are indistinguishable by it, and a matching
number is a coincidence, not an identification.

Then `card_claim.py acquire --name tilerl-tput --cards 2 --pid 3304585 --require-device`.

## What the gap cost

While the claim was absent, any `pod_run.sh` targeting card 2 would have exited 3 (orphan
>64 MiB), and an aupai session looking for orphans would have found one. Nothing was killed,
which is luck: the correct response to an orphan card holding 66 GB is to investigate it, and
the investigation path the alert names leads to an unusable pid.

## Rules

- **A multi-arm wrapper claims per arm.** The claim binds to one process; every arm boundary
  is a window where the ledger reads stale and the card reads orphan. If the launcher exports
  a claim helper, the wrapper has to be something that can call it — not a shell loop.
- **Identify a card's holder from `/proc/<pid>/environ`, never from smi's pid or its byte
  count.** Inside a container smi reports host-namespace pids that no `/proc` lookup resolves;
  `CUDA_VISIBLE_DEVICES` is the positive evidence, and a matching MiB figure is a coincidence
  that looks like proof.
- **Before writing a new wrapper for a shape the tree already runs, grep for the shape.**
  `tp_step_arms.py` had the per-arm claim, the reason, and the failure it prevents.
- **Advice printed by a tool has to agree with what the tool knows.** Both facts lived in
  `card_claim.py` — the measurement at `:201` and the contradicting suggestion at `:1306`.
  Distance inside one file is enough for a recorded finding to stop being enforced, so a
  remedy string is the place to check first when a documented fact keeps getting re-learned.
