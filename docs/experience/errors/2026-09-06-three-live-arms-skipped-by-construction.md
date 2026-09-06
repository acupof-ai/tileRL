# Three live arms skipped by construction, and the skip said only "too short" — cpu, 2026-09-06

> Status: FIXED (the arms fire; a control confirms they still skip honestly).
> Not yet re-run against the V100 — `pending-remote`.

## Context

`scripts/api_e2e.py` gained three stop-sequence arms with
[#174](../wins/2026-09-06-stop-sequences-matched-on-text-not-token-ids.md). On the
V100 run of merge sha `753da30`: **rc 0, 11 passed, and all three stop arms
SKIPPED** with `"reply too short to cut inside"`. The feature's only automated
coverage against real weights never executed. 27 established the behaviour by three
manual probes instead.

## Root cause

The arms derived their stop from the model's own reply to the shared `ask` prompt:

```python
ask = [{"role": "user", "content": "What is 2+2? Answer in one word."}]
...
return text[1:4] if len(text) > 5 else None
```

**`ask` demands a one-word answer, so the reply cannot be long enough to cut
inside.** The skip was guaranteed by construction, on every run, forever. Nothing
about the deployment caused it and no future deployment would fix it.

Two things kept that invisible:

1. **The skip carried no number.** `"reply too short to cut inside"` reads as a
   property of this run — a terse model, a low cap — rather than of the arm. With
   the length printed, `6 chars` against a threshold of 8 is obviously not a
   deployment property.
2. **The skip floor counted it as tolerable.** The floor exists to stop a
   deployment skipping its way to exit 0, and it fired correctly at 3/14 — under a
   third, so `rc 0`. A floor on the *proportion* of skips cannot see that a
   specific arm can never run.

## Fix

The arms get their own prompt (`say_more`, asking for two full sentences), the stop
is cut past the first word, and every skip names its measurement:
`reply too short to cut inside: 6 chars of prose`.

## The control found a second defect in my own arms

With a short canned reply the control was supposed to skip. Instead **two of three
arms passed** — on a stop of `'/thi'`. The byte tokenizer has no `<think>` token, so
the reply still carries a literal `</think>`, and `text[1:4]`-style slicing cut the
stop out of that **markup**. The chat arms then asserted successfully that a reply
was truncated at the closer, which tests nothing about stop sequences; the messages
arm failed with `end_turn`, correctly, because the engine's reasoning gate refuses to
match before the closer.

So the passing arms were the wrong ones. The stop is now taken from
`text.split("</think>")[-1]` — prose only, never across the closer.

My first repair for that was to **refuse** a reply containing the closer, which made
all three arms skip on every canned run: correct-looking, and it would have removed
the local smoke test entirely, leaving the arms exercised nowhere but the pod.

## Controls

| canned reply | arms |
|---|---|
| `"The capital of France is Paris. It has held that role for centuries."` | 3/3 fire — `cut at 'apit'`, stream `stop absent`, `stop_reason=stop_sequence` |
| `"Paris."` | 3/3 skip, each naming `6 chars of prose` |

## Rule

A check that reports SKIP must report the measurement that caused it, and the value
must be capable of changing. A skip guaranteed by the check's own inputs is a dead
arm wearing a pass, and a proportional skip floor cannot detect it — the floor
bounds how many arms skip, never whether a given arm could ever run.
