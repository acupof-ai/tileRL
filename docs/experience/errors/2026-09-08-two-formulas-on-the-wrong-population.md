# Two formulas, each on the wrong population — 2026-09-08

**Status:** both caught. Two unrelated defects in one hour with one shape — a correct procedure
applied to a population it does not describe. The first read card occupancy as card ownership; the
second computed a scheduling idle fraction on a sorted list the scheduler would never see.

## I. The memory row is not the ownership record

Caught by someone else's guard, not by my own check. No run was lost; one OOM'd before being
killed, and a second was one `DEVICE_WAIT` away from writing the same output file as its own
replacement.

### What I did

Launched four shards of a pass@k measurement across cards 1, 2, 3, 7. Before launching I read
`nvidia-smi --query-gpu=index,memory.used` and picked cards that looked quiet.

Two of the four were already someone else's:

- **card 3** held `tilerl-tail6k` (48's, claimed 10:47:31). My shard OOM'd against it —
  `torch.OutOfMemoryError ... Process 834695 has 76.58 GiB in use` — and *then* `pod_run`'s claim
  check refused and killed it: `cards ['3'] are claimed by {'3': ['tilerl-tail6k']}. Queue -- do
  not spill onto a claimed card.`
- **card 2** held `tilerl-arm6k` (claimed 10:51:40). That shard sat polling for a device fd it
  could never get, and would have been killed at `DEVICE_WAIT`.

### The defect

**`nvidia-smi` answers "is this card busy". The claim ledger answers "whose is this card".** Those
are different questions and I substituted the first for the second. A card at 0 MiB can be claimed
by a job still loading weights — which is exactly what cards 4 and 6 looked like minutes later when
they were mine — and a card with memory in use can be an orphan nobody owns. Neither reading tells
you what the other one does.

The tree already records this (`ownership-is-the-ledger-occupancy-is-the-device`), and
`pod_run.sh`'s own header says an unclaimed card is what gets a container restarted under someone
else's run. I read the weaker instrument because it was the one I already had a command for.

### The second, worse half

Relaunching offsets 25 and 50 onto free cards did not retire the shards that were still polling.
The card-2 shard wrote `/work/pk_25.jsonl` — **the same path its own replacement now owned.** Two
writers on one output file, and the file is `open(..., "w")` with a per-row flush, so the result
would have been interleaved rows from two processes that each believed they were alone.

Neither the claim check nor `pod_run`'s live-name guard catches this: the guard is per *job name*
(`pk2` and `pk4` are different names) and the claim is per *card* (2 and 4 are different cards).
**The collision is on the output path, which nothing owns.** Killed the card-2 process by pid after
confirming its `CUDA_VISIBLE_DEVICES=2` at exec time — the card authority — rather than by pattern,
which would have matched the replacement too.

### Rule

**Before launching onto a card, read the ledger, not the memory row.** One command:
`card_claim.py status`. Occupancy and ownership disagree in both directions, and the direction that
costs someone else their run is the one where the card looks free.

**And a relaunch has two halves: start the new one, retire the old one.** I did the first and
treated the guard's eventual kill as the second. It would have been — for the card — while the
output file collided the whole time. When re-pointing work at new hardware, the thing to check is
not only what it will compute on but what it will write to.

See [[ownership-is-the-ledger-occupancy-is-the-device]],
[[a-proxy-substituted-for-the-authority]], [[a-caller-that-claims-a-card-blocks-its-own-job]].

## II. An idle fraction computed on a sorted list

Asked how much of a decode tick goes idle waiting for the longest row in a group, I took the 32
measured completion lengths from the 6144 rerun, sliced them `range(0, n, 8)`, and reported
**15.0%**. That list was **sorted**, so every group held eight rows of similar length and the
group's max sat close to its mean.

**The engine groups by arrival, not by length.** Shuffling the same 32 values:

| grouping | group=8 pooled idle |
|---|---:|
| sorted (what I computed) | 15.0% |
| **random (what the engine sees)** | **40.7%** [p5 35.4, p95 43.9] |

**2.7x, entirely from the order.** And 15.0% is not merely wrong — sorting rows by length before
grouping is one of the *fixes* for tail idle, so I had computed the post-fix number and reported it
as the status quo.

The population matters as much as the order. Those 32 are exactly the problems that hit the 2048
cap, so nothing shorter than 1413 tokens is in them by construction. Adding back the 68 that
terminated (reconstructed from the 2048 run's histogram, so approximate) raises it further:

| population | random grouping, group=8 |
|---|---:|
| long rows only (n=32, min 1413) | 40.7% |
| **full (n=100, min 36)** | **57.1%** [p5 53.3, p95 60.3] |

So an idle figure is uninterpretable without three things beside it, and I got two of them wrong in
one calculation:

```
idle = 1 - Σ len / Σ (max_in_group × group)
  population   which subset -- selection effect worth 16.4 pt here
  grouping     arrival / sorted / random -- worth 2.7x
  pooling      wall-weighted, not a mean of per-step idles
```

**A formula can be right, applied to real measured data, and still answer a different question than
the one asked.** Both errors were invisible in the number: 15.0%, 40.7% and 57.1% are all
plausible idle fractions, and nothing about reading one says which population and ordering produced
it.
