# The "full host" that justified having no DRAM tier is not full

**Date:** 2026-09-03
**Target:** pod (V100, sm70) — one command, no GPU

## Context

`KvTier` (HBM→SSD prefix spill, `e9d5852`) carried a docstring explaining its
own shape:

> On a 32 GB V100 with a **full host** there is no DRAM residency tier, so it is
> HBM→SSD.

ckl asked whether KV offload to DRAM and SSD exists. Answering it meant
checking the premise rather than quoting the comment.

## Root cause

`free -g` on the pod:

```
               total        used        free      shared  buff/cache   available
Mem:              31           5           7           0          19          25
```

**25 GB of 31 available.** The host is not full. The premise was written
2026-09-02 and was either wrong then or has been wrong since; either way the
design rationale in the docstring does not hold, and a reader deciding whether
to add a DRAM tier would have been told the question was already settled.

Note what this does *not* establish: that a DRAM tier is worth building. The
second number that decision needs — SSD reload latency on a prefix hit — is
still unmeasured. What is retired is the stated reason for not considering it.

## Fix

The rationale is gone rather than corrected, because the class went with it: the
rebase onto current `main` found upstream had reworked `PrefixStore` for GDN
state snapshots plus a byte budget (`state_bytes`), which occupies the same seam
as `KvTier`'s eviction hook. Nine conflict hunks in `kv_cache.py` were two
independent designs on one class. Taking upstream's and re-landing the tier on
top is the cheaper path than hand-merging, so KvTier, its CLI flag, its seven
unit tests and its e2e cold-hit gate are all deferred to one tracked item.

What was kept from the branch's side: `state_bytes` derived from free HBM
(`mem_get_info()[0] // 4`) rather than upstream's 8 GiB default, which is most
of a 32 GB V100's post-weights headroom.

## Rule

A comment that justifies a design decision with an environment fact is a claim
with no gate on it, and it ages silently — nothing fails when the host grows.
Re-measure the fact before quoting the rationale it supports; one command
settled this one.
