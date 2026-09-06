# An omitted max_tokens means the context remainder — server, 2026-09-06

> Status: Shipped

## Context

ckl, on the V100 endpoint: "现在最大输出长度是多少有限制吧 默认尽量输出不限制不超上限就行" —
a reply that stops early reads to a client as a dropped stream, and the report was that
replies were being cut.

Two independent caps were producing that, one on each side of the wire:

- **Server:** `/v1/chat/completions` and `/v1/responses` defaulted an omitted cap to a flat
  **512**, so any answer longer than 512 tokens ended at `finish_reason=length`.
- **Client:** the V100 demo page sent `max_tokens` **unconditionally** — `main.ts:30` built
  `{messages, max_tokens: cap, ...}` with `cap = Math.max(1, Number(budget.value) || 512)`
  and markup carrying `value="512"`. tilerl-25 found and fixed this in #194; **the server
  change alone would not have been visible from that page**, because the field was never
  omitted.

This entry is the server half. Both halves are needed: omission semantics are only reachable
by a client that omits the field.

## What Worked

`Engine.room_for(prompt_tokens)` — the largest `max_new_tokens` a prompt can ask for and
still be admitted, returning the two ceilings `submit` already enforces rather than letting
each route re-derive them:

```python
by_total = self.limits.max_total_tokens - prompt_tokens
by_pool = BLOCK_TOKENS * self.usable_blocks - prompt_tokens - self._width + 1
return max(0, min(by_total, by_pool))
```

`by_pool` is the term a caller would omit. It carries `usable_blocks` (net of the captured
tick's padding row) and `- width + 1` (the drafts a verify tick materializes past the last
token), so the default is admitted on a spec-decode engine too rather than 400-ing on the
second ceiling after clearing the first.

**Tight on both sides, measured on two engine shapes at a 40-token prompt** — `room_for(p)`
accepted, `room_for(p) + 1` refused, and the refusal names the bound the shape was built to
hit, so each shape exercises a different term of the `min`:

| shape | `room_for(40)` | `+1` refusal |
|---|---|---|
| `blocks=256, max_ctx=512` | 472 | `request (513 tokens) exceeds max_total_tokens (512)` |
| `blocks=8, max_ctx=4096` | 88 | `request (129 tokens) exceeds KV pool capacity` |

**The route defaults**, only where a client can legally omit the field:

| route | field | omitted ⇒ |
|---|---|---|
| `/v1/chat/completions` | `max_tokens` | `room_for(len(input_ids))` |
| `/v1/responses` | `max_output_tokens` | `room_for(len(input_ids))` |
| `/v1/messages` | `max_tokens` | **unchanged** — Anthropic declares it required, so there is no omitted case |

`is not None` on the chat route, `or` on Responses: 0 is not a usable cap either way, and
Responses' own `ge=1` already rejects it, so the difference is cosmetic.

## The defect this exposed: `DataParallelEngine` did not forward `room_for`

Adding a method to the engine seam means every implementation of that seam needs it. The
wrapper did not have it, so **`serve --devices` returned 500 on every request that omitted
`max_tokens`** — the same defect class as the missing `limits` that 400-ed every Claude Code
turn under `--devices`, one route later.

Fixed at `parallel.py:95-98`, `min` across replicas for the same reason `limits` takes the
min: `submit` routes to the shortest queue, so the only budget every replica honours is the
smallest one's.

**Negative control ran.** Dropping the forwarder and re-running the wrapper test gives
`{'plain': 200, 'plain/omitted': 200, 'wrapped': 200, 'wrapped/omitted': 500}` — red on the
omitted-cap assertion specifically, with every other arm still green, so the control fails
for the reason under test and not by a second route.

## Gates

`413 passed, 14 skipped` on the CPU target, from a clean tree — the first full-suite run was
discarded because it overlapped the control's mutation of `parallel.py` and could not be
trusted.

Four assertions carry the change:

- `test_an_omitted_max_tokens_gets_the_context_remainder` — spies on `server.sampling` and
  asserts `max_new == 4096 - prompt_tokens`, **not** the reply. The tiny model's answer is
  short either way, so a test reading only the reply stays green with the 512 in place.
- `test_an_omitted_max_output_tokens_gets_the_remainder_on_responses` — patches
  `responses.sampling`, a separate import: patching `server.sampling` would not observe it,
  so one route's gate is genuinely not the other's.
- `test_an_explicit_max_tokens_is_still_honoured` — the control against an unconditional
  remainder. Reverting both routes to 512 turns both omitted-case tests red on
  `assert seen[-1] != 512` while this one stays green.
- `test_the_clamp_survives_the_data_parallel_wrapper` — extended with the omitted-cap arm
  through plain and wrapped engines.

The `_ScriptedEngine` double returns a **flat 64** rather than the real formula. Re-deriving
it there would let a broken `Engine.room_for` pass all 40 SDK arms — the double implementing
the behaviour under test.

## Rule

**A method added to a seam is added to every implementation of that seam, and the test doubles
are not the ones that matter.** The 16 SDK failures were loud and harmless — a `getattr`
fallback in the route would have silenced them and left the `--devices` 500 shipping. The
doubles told me the seam had widened; the wrapper was where that actually cost something.

**A cap on the client and a cap on the server produce the same symptom.** "Replies are being
cut" was true on both sides here, and the server fix is unobservable from a page that always
sends the field. Check which side omits before concluding either half worked.

## Results

| date | commit | target | metric | value |
|---|---|---|---|---|
| 2026-09-06 | (this) | cpu | omitted `max_tokens`, chat | `4096 − prompt` (was 512) |
| 2026-09-06 | (this) | cpu | omitted `max_output_tokens`, responses | `4096 − prompt` (was 512) |
| 2026-09-06 | (this) | cpu | `/v1/messages` | unchanged, field required |
| 2026-09-06 | (this) | cpu | `serve --devices`, omitted cap | 500 → 200 |
| 2026-09-06 | (this) | cpu | full suite | 413 passed, 14 skipped |

## Limitations

- **No throughput or latency claim.** This is an admission-arithmetic change; a larger cap
  changes how long a reply may run, not how fast a token is produced.
- **CPU target only.** The remainder is arithmetic over `limits` and the KV pool, with no
  arch-dependent term, but it has not been exercised against the live V100 server. tilerl-25
  will check the end-to-end arm (empty box ⇒ a reply past 512 tokens) after 27 redeploys.
- **The remainder is per-request and can be the pool number, not the context number.** On a
  configuration where the KV pool binds first, `room_for` returns less than
  `max_total_tokens − prompt`. That is correct and is not a truncation bug.
