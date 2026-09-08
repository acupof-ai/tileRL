# The authority order's first rung does not exist on the V100

## Context

A V100 process (pid 3171701, 25978 MiB) had held the card all day with no known owner.
`AGENTS.md` gives the order to resolve that: read `runs/claims/` FIRST, because the ledger
says whose the card is and `nvidia-smi` only says busy — then ppid, then etime.

Executing it: `runs/claims/` does not exist on the V100. Neither does the mechanism it
belongs to. `/work` — the tree every pod rule names — is absent; `find / -maxdepth 4 -name
card_claim.py` returns nothing; `/*/aupai` and `/home/*/aupai` do not match. `pod_run.sh:46`
defaults `AUPAI=/work/aupai`, so the claim helper the rule depends on has no path here. In
the repo, `git ls-files | grep -i claim` finds one docs entry and no `runs/claims/`.

So the rule's first rung is not something to skip on this card. It is not installed.

## Root cause

The rule was written from the H20/aupai pod, where `/work` and the claim ledger exist, and
generalized to "the card" without naming which pod. The V100 is reached by a different alias
and has a different layout: home-directory trees (`~/tilerl-git`, `~/tilerl-v100`), logs in
`~`, no `/work`, no card ledger. An instruction that assumes a filesystem layout fails
silently on a host that does not have it — silently, because the failure mode is a missing
directory, which reads as "no claims recorded" rather than "no claim mechanism".

Those two readings license opposite actions. "No claim recorded" plus "0% utilization" is an
orphan card by `pod_run.sh:97`'s own rule. "No claim mechanism" says the ledger is not
evidence about this card at all, so the only thing an empty result establishes is that the
question cannot be answered here.

## Fix

Report the rung as **not executable on this host**, not as executed-and-empty, and fall
through to what the host does have. What answered the ownership question instead:

- `ps -o args` — a repo script (`serve_v100.sh`), not a stray process.
- `ppid` — 3171691, `bash scripts/serve_v100.sh`, whose own ppid is 1.
- `/proc/uptime` minus `/proc/<pid>/stat[22] / CLK_TCK` — 61182 s = 17h00m, agreeing with
  `lstart` and `date`. Three sources, not a single `etime` reading.
- `/health` — `finished 0` since the current boot, which the log dates to 17 h ago.
- 5 utilization samples 2 s apart — 0% each.

Ownership itself came back **unresolvable from records**: every session on the pod runs as
the same unix user, so the process owner cannot distinguish sessions — the same shape as a
git author not showing a branch owner. The only signal was the tree name: `tilerl-git`
carries no `s-<session>` suffix, so it does not follow the one-tree-per-session convention.
"Records say no one" is a finding; "it belongs to nobody" would not have been.

One mechanism fact the investigation produced that a kill would have needed:
`serve_v100.sh` is a restart loop (trap TERM/INT plus `MAX_RESTARTS`; the log holds 44
`=== boot` against 22 `=== exit`). TERM to the child 3171701 gets it restarted by the
supervisor. Stopping means TERM to 3171691.

## Rule

A rule that names a path names a host. Before reporting a ledger empty, establish the ledger
exists — an absent mechanism and an empty mechanism return the same thing to `ls`, and only
one of them licenses acting on the card. When the first rung of an authority order is not
installed, say so and name the rungs that replaced it; do not report the order as followed.
