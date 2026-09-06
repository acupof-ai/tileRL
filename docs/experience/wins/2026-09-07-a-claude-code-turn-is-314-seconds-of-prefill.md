# A Claude Code turn is 314 s, and 43 of its 46 prefill forwards precede the first token — V100 sm70, 2026-09-07

> Status: Shipped (measurement only; no runtime change)

## Context

Question 3 of the agent-turn series: the per-turn wall clock of a real Claude
Code session against the live V100 endpoint (`09e1e84`, `--max-ctx 32768`,
`--max-batch 1`, depth 1, NVFP4 Qwen3.8-27B). One client, sequential turns,
timed by a forwarding proxy between the client and the endpoint
(`scripts/time_agent_turns.py`) with the pool sampled every 5 s on the pod
(`scripts/sample_pool.py`, cadence chosen deliberately — see Rule).

The task: create `add.py` and `check.py`, run `check.py`, report its output.
The client exited 0 with `ok`, and both files are on disk with the right
contents, so the turns did the work rather than merely returning.

## What Worked

**The measurement, at the second attempt. The first attempt 400'd on turn 1
and that refusal is a finding in its own right.**

### Arm 1 — a turn from this repo does not fit the context

    HTTP 400  request (39472 tokens) exceeds max_total_tokens (32768)
    turn 1, 0.584 s, asked_max_tokens 32000, 2 messages, 29 tools
    0 turns completed

Attributed through the server's own `render_prompt`, tokenized with the
checkpoint's tokenizer on the pod:

| part | tokens | share | origin |
|---|---:|---:|---|
| tools (29 schemas + `<tools>` wrapper) | 18,871 | 47.8% | Claude Code |
| CLAUDE.md + memory reminder | 11,090 | 28.1% | this machine's config |
| session hook (Ponytail) | 7,462 | 18.9% | this machine's config |
| system prompt | 2,039 | 5.2% | Claude Code |
| template + prompt floor | 10 | 0.0% | Claude Code |
| **total** | **39,469** | | vs the server's **39,472** |

**47% of that prompt is this machine's own config, not Claude Code's.** The
remaining 3 tokens are not slack: `engine.py:516` computes
`total = len(tokens) + params.max_new_tokens`, so 39,472 = 39,471 + 1 —
`room_for` returned 0 and #207's `max(1, ...)` passed the request through to be
refused. The clamp behaves as #205/#207 describe; it cannot help when the
prompt alone exceeds the context. **The 400 is correct behaviour.**

### Arm 2 — Claude Code's own turn, with this machine's config out

`CLAUDE_CONFIG_DIR` pointed at an empty dir and a scratch cwd: hook 0 chars,
reminder 306 chars, prompt **21,727 tokens**, fitting with 11,041 to spare.

    3 turns, 3 ok, 0 failed
    seconds per turn:  314.33  |  16.39  |  13.18
    total 343.90 s     median 16.39 s
    client exit 0, `ok` on disk in add.py + check.py

**Turn 1 is 19x turn 3, and the pool table says where the 314 s goes** — not
in decoding:

| t (s) | running | prefill_fwd | decode_fwd | tokens_gen | free_blocks | /health s |
|---:|---:|---:|---:|---:|---:|---:|
| -4.2 | 0 | 3 | 128 | 250 | 2029 | 0.01 |
| 0.8 | 1 | 8 | 128 | 250 | 676 | 13.77 |
| 73.5 | 1 | 26 | 128 | 250 | 688 | 33.97 |
| 194.4 | 1 | 44 | 128 | 250 | 688 | 87.66 |
| 294.2 | 1 | 46 | 129 | **253** | 688 | 6.51 |
| 310.7 | 1 | 46 | 207 | 403 | 678 | 0.12 |
| 354.3 | 0 | 48 | 268 | 520 | 676 | 0.00 |

`tokens_generated` sits at **250 for the first 294 seconds** while
`prefill_forwards` climbs 3 → 46. The first token of turn 1 arrives at
t≈294 s; everything before it is prefill.

**43 chunks is the right observation; "the chunk count is the cost" was the
wrong mechanism, and it is corrected here rather than left standing.**
`max_num_batched_tokens = 512` (`engine.py:183`), and 21,727 / 512 = **43** —
matching the observed +43 forwards before the first token exactly. What that
does not do is distinguish per-chunk cost from per-token cost: `chunks =
ceil(tokens/512)` makes the two collinear, so both models predict this arm.

tilerl-25's sweep (#213) holds the chunk count fixed and separates them:
**544 → 1024 tokens keeps `prefill_forwards` at 2 while TTFT goes 2.57 → 5.16 s
(+101% for +104% tokens)**, repeats at 1024 of 5.16/5.25/5.22, sd 0.04 s.
Per-chunk predicts those two arms are equal; they differ by 2x. **So prefill
cost is per token, and quadratic** —
`ttft = 0.56 + 0.00422 n + 4.117e-07 n²`, R² 0.9999.

The 294 s measured here is the out-of-sample check on that fit: at n=21,727 it
predicts **287 s against 294 s measured, 2.4%**, from a fit whose largest arm
was 9,483 tokens. A linear model predicts 174 s. The n² term is 68% of the cost
at this prompt length, which is also why turns 2 and 3 are 16 s and 13 s — the
prefix cache is not saving 43 ticks, it is saving the quadratic.

**Consequence: `max_num_batched_tokens` is not the lever.** Raising it changes
how the same tokens are grouped, not how many there are. That conclusion is
27's ruling on #213; the target is the prefill kernel.

Two secondary readings from the same rows:

- **Prompt占 66% of the pool.** 21,727 tokens / 16 tokens-per-block = 1,358
  blocks of 2,048; peak `pool_used_blocks` 1,404, min `free_blocks` 644. A
  second concurrent client of this shape does not fit — the queue-and-wait
  work is unaffected by this entry.
- **`/health` median 8.12 s, max 87.66 s during prefill, 0.002 s idle.** Four
  orders of magnitude on the same code, which confirms the contention is the
  engine lock held across `step()` and not a slow handler.
- Prefix cache did work across turns: `prefix_hits` 0 → 2, so turns 2 and 3
  reused turn 1's blocks. That is most of why they are 16 s and 13 s.

## Rule

**A 21.7k-token prompt costs 294 s of prefill before its first token, and that
dominates the first turn of an agent session — 294 s of 314 s with zero tokens
emitted.** Per-turn latency of an agent client is a prefill number, not a decode
number. Decoding was 270 tokens over 140 forwards in the last 60 s.

**The cost is per token and quadratic, not per chunk** (#213), so
`max_num_batched_tokens` is not the lever and the prefill kernel is. I stated
the per-chunk mechanism first, on an arm where the two models are collinear —
`chunks = ceil(tokens/512)` — and it took a same-chunk-count pair to separate
them. **An arm that both hypotheses predict is not evidence for either.**

Three instrument rules this measurement paid for:

- **A char count cannot check a token ceiling.** The first attribution
  tokenized `json.dumps(tools)` and got 30,787 against the server's 39,472 —
  0.76x. The server wraps schemas in `<tools>` with instruction text; only
  `render_prompt` agrees with it (0.9999x).
- **Capture the body from the cwd that will send it.** The same script in a
  tempdir vs this worktree produced CLAUDE.md at 8,518 vs 41,482 chars — 5x
  apart on the part that decides the verdict.
- **A suppression must be checked before it is used.**
  `--settings '{"hooks":{}}'` suppressed **nothing** (hook still 25,896
  chars); only `CLAUDE_CONFIG_DIR` at an empty dir dropped it to 0. Arm 2
  would have reproduced arm 1's 400 and burnt the card.

And two probes that printed success while measuring nothing: a
`--allowedTools` sweep whose three arms all sent 29 tools (the flag gates
permission, not the schema set), and before it the same sweep printing "arm
done" three times having sent zero requests, because `--allowedTools "Read"
"say ok"` consumed the prompt as a second tool value and the client's output
went to `/dev/null`.

Sampler cadence was **5 s, not the 20 s suggested** — a coarse poll cannot
attribute an event, which the sibling stop-repeat measurement had just shown at
2 s. It did not bind here (the turn boundaries are tens of seconds apart), but
both instruments now stamp an absolute epoch per row, because a cadence index
cannot say which turn a sample fell in.

## Results

| date | commit | machine | target | model | prompt tok | turn 1 s | turns 2-3 s | prefill fwd before tok 1 |
|---|---|---|---|---|---:|---:|---:|---:|
| 2026-09-07 | 09e1e84 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 21,727 | 314.33 | 16.39 / 13.18 | 43 |
| 2026-09-07 | 09e1e84 | V100 32GB | cuda sm70 | Qwen3.8-27B NVFP4 | 39,471 | — (400) | — | — |

Serve child pid 2855749, stamp `/data00/home/chenkailun.c/tilerl-git`
starttime 402881162, identical at start and end of both arms.

Raw artifacts: `/tmp/q3b/turns.jsonl` (proxy rows), `/tmp/q3b/pool.jsonl`
(27 samples), `/tmp/q3/turns.jsonl` (arm 1's 400).
