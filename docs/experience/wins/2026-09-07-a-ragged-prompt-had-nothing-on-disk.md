# A ragged prompt had nothing on disk a client could match — H20 card 1, 2026-09-07

> Status: Shipped

## Context

The SSD prefix tier spilled 321 MiB per conversation and every bench arm read
0 hits back. Two separate causes; this entry is the write side. The read side --
a row admitted before its own prefetch lands -- is a separate fix, and the
measurements below were taken with both in place.

`_finish_prefills` published the prompt-only entry only when
`prompt_len % BLOCK_TOKENS == 0`. Prompt lengths are whatever the chat template
renders to, so **15 of 16 prompts never reached that branch**. The one entry that
did reach disk was the DECODE publish — `req.tokens[:materialized]`, prompt plus
the reply the model generated. A replayed turn 2 cannot reproduce it:
`blocks_to_text` (`prompt.py:65`) strips reasoning from replayed history by design,
and the prompt tail contributes `<think>\n`.

So the tier held exactly one entry per conversation and it was the one entry no
client could ever match.

## What Worked

Two changes, one rule. `_pick` cuts the final prefill chunk back to a block
boundary, so a boundary publish exists for any prompt length; `_finish_prefills`
spills exactly that boundary and no earlier one. Both ask
`_last_prefill_boundary(n)` rather than each deciding for itself — they disagreed
in the first draft, and the symptom was that `len % 16 == 1` published an entry
and then withheld it from disk.

A 1-token tail cannot be its own chunk: `width` bucket-rounds only when
`chunk > 1`, so a T=1 prefill row reaches the kernels with a zero block size and
raises `Divide by zero` from `T.ceildiv`. The cut backs off one block there,
costing 16 tokens of prefix not served on a hit.

Measured on card 1, 2729-token prompt, against turn 2's 2755 ids:

```
before   entries [2736]        2736 diverges at 2727     servable: none
after    entries [2720, 2736]  2720 prefix true          servable: [2720]
```

> **Provenance (2026-09-10 cleanup):** the one-off `scripts/probe_ssd_read_miss.py` was deleted here; rerun it by hand with `scripts/pod_run.sh ssdprobe 1 -- /work/tl013/bin/python -u scripts/probe_ssd_read_miss.py`. No code replaces the instrument; this entry is the record, rebuild on a card from the command.

Cost: one extra boundary spill per prefill on top of the decode spill
(`ssd_offered` 1 → 2 on this prompt), plus one forward for the ≤17-token tail.
**The wall-clock cost is not isolated** — the long arms differ in what they recover
as well as in what they write, so the difference is not attributable to the extra
write. A write-through before/after arm would settle it and was not run.

Two instrument defects were found and fixed on the way, both of which made a bad
number look like a good one. `_prefix_check` read the largest `.kv`, which is always
the unservable decode entry, and reported DIVERGES over a tier holding a working
entry. `_matched_tokens` inferred coverage from file-size ratios against a 512-token
unit calibrated when six entries formed a ladder; with two entries it returned 512
for a hit covering 2720, and the ceiling built on it claimed the arm saved 2.8x more
prefill than it could have. The guard that exists for exactly that condition sat
below a branch which matches on every warm restart, so it never ran.

## Rule

A publish that fires on an alignment condition serves only the inputs that happen
to meet it. If the condition is `% BLOCK_TOKENS == 0` and the input is a token
count nobody controls, the feature is off for 15 of 16 requests and every counter
still reads healthy.

Where two sites must agree on the same boundary, give them one function to ask.
The failure when they drift is not a crash but a hole at one residue class.

A guard placed in an `elif` chain below a branch that matches the normal case never
runs. Both of this bench's sanity checks were written after a bad number and both
were unreachable or reading the wrong file by the time they were needed.

## Results

Six runs of one sha on card 1, 2729-token prompt, `--gen 8`:

| run | probe MiB/s | cold s | faulted s | control s | control/faulted |
|---|---:|---:|---:|---:|---:|
| row58 | 150 | 2.584 | 1.234 | 2.519 | 2.041 |
| rep1 | 161 | 2.531 | 1.221 | 2.517 | 2.061 |
| rep2 | 180 | 2.558 | 1.288 | 2.637 | 2.047 |
| rep3 | 147 | 2.725 | 1.266 | 2.455 | 1.939 |
| stinstr | 297 | 2.601 | 1.260 | 2.515 | 1.996 |
| row58c | 268 | 2.512 | 1.803 | 2.585 | 1.434 |

**2.041x, range 1.939–2.061 over five runs.** The sixth (row58c, 1.434x) is a single
outlier whose cause is not established. The eviction probe looked like the
explanation on the first four runs — the slow run had the highest rate — but
`stinstr` then ran at a higher probe still (297 against 268) and landed in the main
cluster, so the probe does not separate them and that hypothesis is dropped rather
than reported. One run in six sits 27% low for a reason nobody has found.

**Host reboot / evicted cache: composed, not measured as an arm.** `DONTNEED` does
not fully evict (it only drops pages nothing else references), so the bench composes
this from a standalone read at the device's measured bandwidth plus the prefill of
the tokens the hit did not cover, rather than from a partially-evicted arm.

**Where the faulted arm's 1.260 s goes** (measured on `stinstr`, the only run
carrying both timers):

```
kv fetch      122 ms   reader thread, off the tick; B = 1393.4 MiB/s (page-cache warm)
st read       151 ms   load_state, on the CALLER at lookup time, not prefetched
remainder     987 ms   tail prefill and the rest of the request
```

The `.st` is 157 MiB against the `.kv`'s 178, and it is in neither `fetch_ms` nor
`fetch_bytes` — so B describes the KV plane only, and the larger of the two reads is
the one that is synchronous. Row 64 moves it onto the reader thread.

Raw artifacts: `/work/row58.log`, `/work/rep{1,2,3}.log`, `/work/stinstr.log`,
`/work/row58c.log` (card 1), `scripts/bench_ssd_restart.py`.
