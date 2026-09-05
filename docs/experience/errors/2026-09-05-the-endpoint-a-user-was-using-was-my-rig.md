# The endpoint a user was using was my measurement rig — 2026-09-05

> Status: fixed; the endpoint is no longer a rig

## Context

ckl reported the V100 endpoint as "not redeployed, output tokens still few." I
checked the deployment and reported three facts: `--max-ctx 32768` present,
`blocks_total 2048`, deployed sha `5bfbfe9` (five commits behind main). I
concluded the flags were current and the short output was the model's own choice,
since a probe I sent returned 1134 tokens with `end_turn`.

Every one of those facts was true. All of them were about a server **I had been
restarting all afternoon**, running **my uncommitted code**.

A peer looked at the same pod and read what I had not:

| observation | value |
|---|---|
| deployed tree | `5bfbfe9` **dirty 5** |
| dirty files | `src/tilerl/engine.py`, `src/tilerl/kv_cache.py` — my unmerged DRAM tier |
| untracked | four `scripts/bench_chat_*.py` of mine |
| `/health` | `dram_demotions: 108`, `dram_demote_ms: 5531` |
| supervisor log | `boot 4`, `5`, `6` within one hour — my restarts |

## Root Cause

I treated the only serving endpoint as a measurement rig because it was the only
GPU I had, and the standing rule I was following — *one GPU job at a time, check
`nvidia-smi` before launching* — is about **contention for the card**, not about
**who is using the service**. `nvidia-smi` showed one process, so every check I
ran said the pod was free. The card was free. The endpoint was not.

Six restarts, a tree carrying unreviewed code, and a `dram_bytes=4 GiB` default
that I later measured as **1.51x worse wall clock** — that is what ckl's requests
were being served by.

The diagnosis I sent had the same blind spot. I read `git rev-parse HEAD` and
reported the sha; I did not run `git status`, so "five commits behind" went out
while "and dirty with your own unmerged tier" did not. The five missing commits
are docs, RL-path, bench and MATH work — **none of them touch the serving output
path**, so my headline finding could not have explained the symptom either way.

## Fix

Deployed tree restored: `git checkout --` the two files (verified first that
`git stash list` was empty and that both files' content already existed on my
branch — `kv_cache.py` md5-identical, `engine.py` an older revision of mine I
backed up before discarding), removed the four bench scripts, advanced to
`origin/main` at `f69806f`, `git status` **0 lines**.

Restarted by pid, not pattern: TERM to supervisor `2187548` and server
`2221393`, card verified at **0 MiB**, relaunched through
`scripts/serve_v100.sh`. The new supervisor reparented to PID 1 and survived the
launching shell's exit, `--max-ctx 32768` confirmed in `/proc/<pid>/cmdline`.

**The endpoint is no longer a measurement rig.** sm70 measurements go to the H20's
card 6 (the tier code is target-independent), or to this card only when ckl is
known not to be using it and a peer has agreed.

## Rule

"Is the GPU free?" and "is anyone using the service?" are different questions, and
`nvidia-smi` only answers the first. Before restarting or reconfiguring a serving
process, check `git status` on its tree and its request log for recent traffic —
a resident server with one process and a clean `nvidia-smi` can still be a user's
endpoint mid-session.

And when reporting on a deployment, `git rev-parse HEAD` is half the state. The
sha names what was checked out; only `git status` says whether that is what is
running. I applied that pairing to my own worktree all afternoon and then omitted
it from the one report where a user was waiting on the answer.
