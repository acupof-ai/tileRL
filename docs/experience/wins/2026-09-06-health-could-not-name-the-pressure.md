# Three of four probe rounds asked a question `/health` had already answered — H20 card 6, 2026-09-06

> Status: Shipped (one stats key). The 2/8/12 wall-clock sweep is unblocked by the
> arithmetic here and still unrun.

## Context

`--dram-bytes` shipped in #149 with a gate proving the flag reaches the tier. The
first wall-clock arm then came back with the tier having done nothing:

```
on   compiles=4  demote=0  promote=0  evict=27  median=2.125
off  compiles=0  demote=0  promote=0  evict=27  median=1.476
off  compiles=0  demote=0  promote=0  evict=27  median=1.506
on   compiles=0  demote=0  promote=0  evict=27  median=1.491
```

`INVALID` on the 4 compiles in the first arm only — `on_1` at 1.491 s sits inside
`off`'s 1.476/1.506, so the apparent 0.825x was first-position JIT, which is what the
alternating arm order exists to expose. But `demote=0` in all four arms with 27
evictions each is not a warm-up problem, and finding out why took four probe rounds.

## What Worked

**The binding operand is blocks, and `pool_used_blocks` said so in round one.**

```
pool free blocks (of 512)  442 → 377 → 234 → 74 → (evict 18) 99 → (evict 9) 129
prefix_state_bytes         598 → 1047 → 1945 → 2843 MiB
prefix_state_bytes_budget  17.68 GiB, flat
```

The pool drained to **74 free of 512** and then evicted. State bytes peaked at
**2.78 GiB of a 17.68 GiB budget — 15.7%**, never a trigger. So the pressure is
BLOCKS, through `evict_until_free`, which calls `_evict_one` directly and by design
cannot demote: a snapshot tier hands back no blocks. **`demote=0` is correct
behaviour, not a defect.**

(The free series above is what the log holds. `pool_used_blocks`, the key that was
already published, is the same quantity complemented — 438 used at the trough, ±1
depending on whether the captured tick's pad block is in `num_blocks`, which the log
does not fix. The complement is the reading; the exact pad is not load-bearing here.)

**Three of the four rounds were self-inflicted.** The probe printed a hand-listed
`keys` tuple, and `pool_used_blocks` — already on main at `engine.py:677`, since
`used_blocks = num_blocks - len(_free)` — was not in it. Measured: `grep -c
pool_used` on round one's log is **0**. The key was in every `/health` response from
the first request; the probe reported a partition of the data instead of the data. It
now prints `sorted(st)`.

I also added a `pool_free_blocks` key before noticing this, and **removed it again**:
it is `blocks_total + 1 - pool_used_blocks`, so it was a third name for a number the
response already carried twice.

**Two wrong readings, both published as findings before being measured.** First I read
`evict_until_free` plus the `--max-ctx 8192` → 512-block arithmetic and concluded
blocks — the right answer, but arrived at by reading code, which is a hypothesis. Then
`blocks_used=0/512` appeared to refute it and I said blocks were excluded; that was
wrong, because `blocks_used` is the ENGINE's counter over blocks live requests own,
and the store's retained blocks are in neither it nor any request.

## The one key that was actually missing

`prefix_state_bytes_budget`. `state_bytes` is set from `mem_get_info()[0] // 4` at
build time, so "is the store at its byte ceiling" was unanswerable from outside at any
number of probe rounds. It is also what resolved a contradiction: using the V100's
measured 1697 MiB budget, the pool only needs `--max-ctx > 1987` for state to bind
first, yet this arm at 8192 was block-bound. The H20's budget is **17.68 GiB, 11.5x
the V100's**, because the card is 96 GiB. A number from one card had been substituted
into another card's condition.

## What this unblocks, and the number I cannot give

The condition for the sweep to measure anything is that state bytes reach their budget
before the pool runs out of blocks. One half is measured: the budget holds
**17.68 GiB / 149.6 MiB = 121 snapshots**, where 149.6 MiB per entry matches the design
page's constant-snapshot finding.

The other half has no single value. Blocks per entry is **not a rate**:

| turn | prompt tokens | free-block drop | entries | blocks/entry |
|---:|---:|---:|---:|---:|
| 1 | 1011 | 65 | 3 | 21.67 |
| 2 | 3054 | 143 | 6 | 23.83 |
| 3 | 6097 | 160 | 6 | 26.67 |

It rises with the prompt, because an entry's blocks are set by the prefix it covers —
`d_free = 65` against `ceil(1011/16) = 64` for that turn's own prompt. An earlier draft
of this entry multiplied the 23.05 average by 121 entries and reported that state binds
at **44,624 tokens of pool**. That number is wrong and is withdrawn: the 121st entry is
far longer than the mean, so the average understates the requirement, and the direction
of the error is toward "the sweep is cheaper than it is".

What is measured, with no extrapolation: at 2 sessions and an 8192-token ceiling,
**19 entries retained 438 of 512 blocks — the pool was 86% full** while state bytes sat
at 15.7% of budget. So the pool has to grow by more than 5x, and the exact figure needs
a run at the candidate `--max-ctx` reading both operands, not arithmetic over an average
that is not constant.

## Rule

Print every field the instrument returns, not the fields the hypothesis names. A
chosen key set makes the probe's output a restatement of what you already believed,
and the failure is invisible: the data was complete and the report was not. Here it
cost three of four rounds and produced one redundant key on the way.

Second: an average is not a rate until you have checked it against the thing it
averages. 23.05 blocks per entry was arithmetically correct and useless as a
multiplier, because the quantity rises monotonically with prompt length — and the
error ran toward the cheaper conclusion, which is the direction that does not get
questioned.

Third: a counter whose ceiling is derived from free memory is per-card. A threshold
measured on one card reads like a constant and is not one.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---|---:|---:|
| 2026-09-06 | pending | H20 card 6 | cuda | qwen38-27b | n/a | n/a | n/a |

No perf surface: one integer read in `stats()`, once per `/health`. `50 passed` on cpu
for `test_kv` + `test_server`, ruff clean.

Three negative controls, each red on its own assertion: dropping the budget key gives
"must both be published or pressure is unreadable"; publishing it as 0 gives
`prefix_state_bytes_budget=0 is not a bound  assert 0 > 0`; and the gate asserts
`pool_used_blocks >= blocks_used`, since reading those two as one quantity is the
mistake that made an idle-looking pool hold 438 retained blocks.

Still unmeasured: wall clock per turn at 2 / 8 / 12 sessions, tier off against on, at a
pool sized so state binds. That is the number `--dram-bytes` exists for and this entry
does not claim it.
