# A lint command whose name matched CI's and whose scope did not

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

I reported PR #307 as ready: "510 passed, ruff clean, one order-dependent tier test that goes
green alone and in its own module — not my diff." A peer went to merge it and found **both
gates red at 20 seconds**, failed on lint, with no test in either gate ever executed:

```
scripts/probe_svd_cost.py:32
  E402  Module level import not at top of file
  I001  Import block is un-sorted or un-formatted
```

A `from collections import Counter` sitting after line 31's comment.

## Root cause

I ran `uv run ruff check <the files I touched>`. CI runs `uv run ruff check`. The offending
import was in a file I had written two commits earlier and was no longer passing as an
argument, so my command could not see it and never would.

The failure is specifically hard to notice because **the commands have the same name**. Every
other instance of this class today involved two visibly different things — a host tensor versus
a card tensor, `.float().cpu()` versus `.cpu().float()`. Here the local step and the gate step
are the same tool, the same subcommand, the same output format ("All checks passed!"), differing
only in an argument list that is invisible in the result.

And the report inherited the error. "ruff clean" was true of what I ran and false of what
blocks the merge; "510 passed" described a run that CI never reached. I then spent a paragraph
of the report characterizing the order-dependent tier test — real, worth investigating, and not
capable of blocking anything, because lint stopped the pipeline 20 seconds before any test
started.

## Fix

`from collections import Counter` moved to the import block; `uv run ruff check` with no path
passes over the whole tree.

The durable part is the reporting rule: **before stating a PR's status, read `gh pr checks
<n>`**, which reads the thing that actually gates the merge, rather than a local command whose
scope I chose. One line, and it is the only check whose object is the same as the gate's.

## Rule

A local verification is a claim about its own scope, not about the gate. When the local command
and the gate command share a name, the argument list is the whole difference and it does not
appear in either output — so match the gate's invocation exactly, or read the gate.

Corollary for reports: a green from a narrower run is not evidence about the wider one, and
detail about a lower-priority failure is misleading when a higher-priority one aborted the run
before it. Say what the gate says.
