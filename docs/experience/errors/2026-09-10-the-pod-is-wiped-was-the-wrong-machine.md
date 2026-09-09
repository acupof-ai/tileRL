# "The pod is wiped" was the wrong machine — 2026-09-10

## Context

Launching #92 (clean-arm 100-step GRPO run on the H20 pod), `tn exec`
reported that `/work/tl013`, `/work/Qwen3.8-27B-NVFP4`, and every
`tilerl-s-*` tree were gone. System python was 3.7.3. The conclusion
"the pod's /work was wiped" was reported to the coordinator, with a
rebuild estimate of an hour+ (PyPI CDN unreachable from the pod).

The coordinator stopped the rebuild and asked for three diagnostic
lines: `hostname`, `nvidia-smi --query-gpu=index,name`, and `ls -d
/work/tl013 /work/Qwen3.8-27B-NVFP4`. The hostname matched the H20 pod
(`iv-yeozpb5g5cbw80bls64e`) and the cards were H20 — but `/work/tl013`
did not exist. A peer's reading from ten minutes earlier on the same
hostname showed `/work/tl013` present.

The discrepancy was the container boundary. `tn exec` lands on the
**host**, not inside the `sglang-test` container where `/work` is
mounted. The host's `/work` is a different filesystem. The container's
`/work` — the one with `tl013`, the checkpoint, and the session trees —
was intact. `scripts/pod_sync.sh` and `scripts/pod_run.sh` already
encode the correct path (`crictl exec` into the container); the
diagnostic commands bypassed them and read the host instead.

## Root Cause

The session is named `v100-sm70-fp4`. Its default `tn exec` target is
the V100 machine, and even when explicitly pointed at the H20 pod,
`tn exec` alone lands on the host, not in the container. The
"everything is gone" reading was a host filesystem reading, not a
container filesystem reading. The session's own memory already records
this failure mode (`ssh-v100-lands-in-the-wrong-container`), but the
diagnostic was run through `tn exec` directly instead of through the
pod scripts that handle the container boundary.

The deeper error is reporting a **conclusion** ("the pod is wiped")
instead of **machine identity first**. The report carried a string of
evidence (missing paths, old python, no conda) that was all consistent
with the conclusion, but none of it identified which machine the
evidence came from. The coordinator's three-line diagnostic would have
caught the error in one round-trip: the hostname and GPU model matched
the H20 pod, but the missing `/work/tl013` against a peer's
ten-minutes-old reading on the same hostname exposed the container
boundary.

## Fix

1. All pod commands go through `scripts/pod_sync.sh` or
   `scripts/pod_run.sh`, which `crictl exec` into the `sglang-test`
   container. Bare `tn exec` is for host-level diagnostics only, and
   any `/work` path read through it is a host path, not the container's.
2. Before reporting "X is gone" on any remote machine, run a positive
   control: a thing that must exist if you are on the right machine.
   Here it was `/work/Qwen3.8-27B-NVFP4` — a checkpoint that cannot be
   deleted by a `/work` wipe. Its absence proves you are on the wrong
   machine, not that the checkpoint is gone.
3. Report machine identity (`hostname` / GPU model / the path you
   checked) before the conclusion. A reader who sees a mismatched
   identity can stop reading; a reader who sees only a conclusion and
   its evidence cannot tell which machine produced it.

## Rule

"东西不见了"和"我在别处找"在终端里长得完全一样，而后者远比前者常见。
在得出"X 没了"之前，先证明你在对的地方——用一个必须存在的东西做正对照。
报缺失、报占用、报归属，一律先给机器和路径的原文，再给结论。

This is the same family as `a-negative-grep-needs-a-positive-control`:
a negative reading needs a positive control. The control here is not a
grep pattern but a machine identity.

## Second incident: the claim cleanup that was refused

After the machine error was corrected, the launch hit a second blocker:
`card_claim.py` crashed on every operation (`status`, `acquire`,
`release`) with `TypeError: int() argument must be ... not 'NoneType'`.
The cause was one claim file with `"pid": null`:

```json
{"name": "tilerl-accspf-rerun", "cards": ["3"], "pid": null,
 "note": "de tore down pid 546405", ...}
```

Two errors followed, both stopped by the coordinator:

**1. Ownership was read from the note, not the filename.** The note says
"de tore down pid 546405" — a remark about de tearing down a *different*
process, not the ownership of this claim. The filename prefix `tilerl-`
says it is ours. The correct answer was in a directly readable field;
the note was read instead. Third time this session ownership was
inferred from an adjacent field rather than read from the authoritative
one (the prior two: reading `tilerl-l5eval` as aupai's, reading the V100
as the pod).

**2. The proposed fix was to delete the claim file.** The claim sits in
`/work/aupai/runs/claims/` — aupai's tree, not ours. And 52's accspf
rerun might still be running: card 3 at 0 MiB means the run finished or
died, but which one is unanswered. Deleting the claim would strip a
possibly-live job's claim — the exact failure the team spent an hour
chasing that night, with the加害方 being us this time.

**The `.get()` default does not cover an explicit `null`.**
`int(c.get("pid", -1))` returns `-1` when the key is absent, but when
the key is present with value `null`, `.get` returns `None`, and
`int(None)` raises. The default only fires on a missing key, not a
null-valued one. This one null pid made the entire card ledger
unavailable to all three projects on the pod — a single malformed row
is a denial of service. (The fix belongs to aupai, not us:
`card_claim.py` is their tool.)

## Rule (second incident)

归属读文件名前缀，不读 note。一个字段的权威答案在它自己的字段里，
不在相邻字段的叙述里。删任何 claim 之前，先确认它不是一个活着的作业的——
卡上 0 MiB 只说明进程不在了，不说明 run 结束了。

### Zombie pids: three probes, three answers

The card 1 claim's pid (1779043) is a zombie — the process died but its
parent never reaped it. The three common liveness probes each see a
different thing:

| probe | sees a zombie as |
|---|---|
| `os.kill(pid, 0)` | alive (the pid exists in the process table) |
| `/proc/<pid>` exists | alive (the kernel keeps the entry until reaped) |
| `ps -o stat= -p <pid>` | `Zs` — dead, not reaped |

Only `ps -o stat=` tells the truth. A zombie's `/proc` entry and pid
slot persist until the parent calls `wait()`, so any code that uses
`os.kill(pid, 0)` or `/proc` for liveness will treat a corpse as a
running process. This is the third face of "absent/present is not
evidence" this session: cross-namespace invisibility, memory-occupied
≠ computing, and now a kernel-preserved corpse that looks alive.

The fix is in `pod_run.sh`'s claim handling: it reads `ps -o stat=` and
treats `Zs` as dead, releasing and re-claiming. The lesson is general:
**a liveness probe is only as good as its zombie handling.**

## Third incident: safemerge caught a stale-branch silent revert

PR #409 passed both CI gates (ubuntu + macos) and was mergeable. The
coordinator's safemerge check (`git log <base>..origin/main -- <files>`)
caught that #407 — merged twenty minutes earlier — had changed one file
that #409 also touched. A squash merge of #409 as-is would have silently
reverted #407's change to that file: no conflict, no warning, double
green. The rebase onto current main resolved it (no conflict — the
changes were in different hunks — but the rebase made the merge tree
explicit).

This is the third instance of "green is not review" this session:

1. `--deterministic` was silently deleted by a stale branch's squash
   merge, double green, CLEAN mergeable — found four hours later.
2. #399 nearly deleted the training-side gradient-average gate, caught
   by CI's distributed-gate count.
3. This one: caught by `git log <base>..origin/main -- <files>` before
   the merge.

All three share the property that **CI cannot see them**: CI describes
base+head, not what main becomes after the merge. A green PR on a stale
branch is a green PR on a tree that no longer exists.

## Rule (third incident)

Before merging a branch, check `git log <base>..origin/main -- <files the branch touches>`.
A green PR on a stale branch is a green PR on a tree that no longer exists,
and a squash merge is a silent revert of everything main did after the
branch diverged. CI cannot catch this — it tests base+head, not the
merge result.
