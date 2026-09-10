# The measurement and the code diverged inside one PR — 2026-09-07

## Context

The SSD prefix tier's headline number, `1.738–1.821x` on a process restart, was
measured on H20 card 6 on 2026-09-05 and recorded in
[wins/2026-09-05-the-ssd-tiers-read-path-did-not-exist.md](../wins/2026-09-05-the-ssd-tiers-read-path-did-not-exist.md).
Two days later the first card run since then came back with the bench's own guard
tripped:

```
faulted  wall 2.070s  ssd_hits 0  entries 1  recovered 1  prefetches 0
"INVALID": "the faulted arm took 0 SSD hits with 1 entries recovered"
```

> **Provenance (2026-09-10 cleanup):** the one-off `scripts/probe_ssd_read_miss.py` was deleted here; rerun it by hand with `scripts/pod_run.sh ssdprobe 1 -- /work/tl013/bin/python -u scripts/probe_ssd_read_miss.py`. No code replaces the instrument; this entry is the record, rebuild on a card from the command.

Re-running at the merged baseline `0356a67` gave the same thing: 0 hits.

## What actually happened

The original artifact was still on the pod (`/work/ssdr6.log`, Sep 5 08:17):

```
faulted  wall 1.706  ssd_hits 1  entries 6  recovered 6  prefix_hits 1
matched_tokens 2560  speedup_faulted_over_cold 1.784
```

**The number was honestly measured.** A real hit, on a real fault-in. The operand
that changed is `entries`: **6 then, 1 now.**

The cause is `9556b2f`, "write-through 45.3% -> 8.96%, scaffolding removed", on the
same branch:

| time | commit | what |
|---|---|---|
| 16:06 | `eae658e` | the measurement: 6 entries on disk, hit at 2560 tokens |
| 18:07 | `9556b2f` | `spill=False` on mid-chunk publishes — 1 entry on disk |
| 19:21 | `0356a67` | both squashed and merged as #128 |

`spill=False` is correct and is the entire 8.96%: six publishes of a 2729-token
prompt spilled 1624 MB to serve one 325 MB entry, and the prompt-complete publish
covers the same tokens. It also removed every entry the bench could hit. The
surviving entry is prompt **plus the reply the model generated** (`_publish_prefix`
writes `req.tokens[:materialized]` during decode, and `req.tokens` is prompt +
output), so it is 2736 tokens against a 2729-token prompt. The bench's turn 2 was
`prompt + " " + followup` with no reply between, which diverges from the stored
sequence at the first generated token and cannot match at any length.

Nobody re-ran the bench between 18:07 and the merge. Neither commit is a defect.

## The disk layout is deliberate, and this is what it costs

One entry per conversation, at the prompt-complete boundary, is the shape that buys
the 8.96%. The cost is now stated rather than discovered: **a second turn that
branches before the assistant's reply gets no hit.** Editing your last message,
retrying with different sampling, or any client that resends the prompt without the
reply misses the tier entirely. A normal chat turn — which carries the reply in its
history — hits.

The bench now sends user / assistant / user, because that is the only shape the
stored entry can serve.

**Superseded in part.** The prompt-complete boundary only exists when
`len(prompt) % BLOCK_TOKENS == 0`, which 15 of 16 prompts fail — so for almost every
real prompt the *only* disk entry was the decode one, and the paragraph above
described a layout the engine did not produce. Row 60 cuts the last prefill chunk to
a block boundary and spills that, so a prompt-only entry now exists at any length:
measured on card 1, entry 2720 is a clean prefix of turn 2's 2755 ids where
previously only the 2736 decode entry existed (diverges at 2727).
[wins/2026-09-07-a-ragged-prompt-had-nothing-on-disk.md](../wins/2026-09-07-a-ragged-prompt-had-nothing-on-disk.md).

## Also fixed here

- The wins entry's Results row named `eae658e`, a commit on `feat/kv-tier-ssd` that
  is not reachable from main (`git merge-base --is-ancestor` says no). Corrected to
  the squash, `0356a67`.
- `ssd_hits > 0` is an assertion with a nonzero exit, not a printed remark. The
  script previously printed `INVALID` and returned 0, so a launcher reading `rc`
  could not tell a bench that measured nothing from one that passed.
- The below-break-even arms were skipped in silence when `n* == 0` (the guard is
  `0 < n_star`, and an unmeasured tier answers 0). They now print `SKIPPED` with the
  reason.

## Rule

**The bench runs on the merge candidate, not on the commit the number came from.**
A default flip or an optimization invalidates every measurement taken before it in
the same PR — cheap to re-check, expensive to discover two days later.

This is the fourth instance of running the gate before the change it is meant to
gate. The write side passing is what hid it: 321 MiB still spilled and the entry
still recovered, so every green half looked like a green whole.

Related: [errors/2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md](2026-09-05-the-ssd-benchmark-never-touched-the-ssd.md).
