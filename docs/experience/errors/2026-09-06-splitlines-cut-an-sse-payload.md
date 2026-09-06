# splitlines() cut an SSE payload, and the count that should have caught it agreed — 2026-09-06

## Context

#158's ubuntu gate failed with
`json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 145`
in `test_completion_stream`, while macos passed and the same tree was green
locally. A rerun of the same sha passed both. That made it intermittent, not
mine — but "intermittent" is not a mechanism, and a rerun identifies nothing.

## Root Cause

**`str.splitlines()` splits on nine separators beyond `\n`; three of them survive
the SSE writer.** `server.py:93` is
`f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"`, so a non-ASCII character
in a delta reaches the wire verbatim. Measured, one arm per separator:

| separator | `splitlines()` splits | `split("\n")` splits | breaks the parse |
|---|---|---|---|
| `\v` `\f` `\r` `\x1c` `\x1d` `\x1e` | yes | no | **no** |
| `\x85` ` ` ` ` | yes | no | **yes** |

The six that do not break it are ASCII controls, which `json.dumps` escapes to
`` and friends — no literal byte is ever emitted. The three that do are
non-ASCII, and `ensure_ascii=False` passes them through: measured
`json.dumps({'c': 'a\x85b'}, ensure_ascii=False)` contains the literal character.

**The characters are reachable from sampled ids, so the input is not exotic.**
`_ByteTokenizer.decode` is `bytes(...).decode("utf-8", errors="replace")` and maps
byte→id+3, so any sampled sequence forming valid UTF-8 decodes to that codepoint.
U+0085 is bytes 194,133 → ids 197,136; U+2028 and U+2029 are three-byte sequences
→ ids 229,131,171 and 229,131,172. All six ids are inside `tiny()`'s vocab of 320,
so **3/3 are reachable**. That is why the failure is intermittent and
platform-dependent rather than deterministic: it needs one unlucky adjacent pair.

Not claimed: a rate. Uniform-draw arithmetic gives p≈1.5e-05 per adjacent pair and
≈2.3e-04 over a 16-token reply, but the model is not uniform, so that bounds
nothing and is not the observed frequency.

**Four call sites, one defect.** `tests/test_server.py:257`, `:356`, `:402` and the
one that failed, plus `scripts/probe_sse_deltas.py:39`. Two use the deterministic
`_TextTokenizer` whose `PATTERN` happens to exclude these characters; `:402` and
the failing one use the `client` fixture's `_ByteTokenizer` and are exposed.

## The guard I wrote against a vacuous test was itself vacuous

The regression test needs the separator to actually reach the wire, or it passes by
absence. My first guard was
`assert len(resp.text.split("\n")) < len(resp.text.splitlines())` — and it **failed
on a stream that does carry `\x85`**, 11 segments against 11.

Cutting at `\x85` splits one frame into a `data: …"content": "a` half and a
remainder that no longer starts with `data:`. The remainder is then filtered out by
the very `startswith("data:")` the parse uses, so the totals match while the
surviving frame is truncated. **A count over a filtered set does not see a cut that
moves an item out of the filter.** The control now parses instead of counting:
`pytest.raises(json.JSONDecodeError)` around the old expression, on the same
response, which cannot agree by construction.

## A measurement I nearly reported wrong

Between those two states I read the segment counts as evidence the character was
being escaped or re-encoded in transit, and tested `latin-1` and a UTF-8
round-trip looking for the transform. There was none: the counts were equal for the
reason above, and the character was on the wire the whole time. Two probes said
"3 vs 3" and "11 vs 11" and I read the second as contradicting the first, when both
were correct and only my metric was wrong.

## Fix

`split("\n")` at all four parse sites — SSE frames are `\n`-delimited by the spec,
so this is the correct reader independent of which separator triggered it — plus
`test_sse_frames_survive_a_separator_splitlines_cuts_on`, which drives a tokenizer
whose fixed text contains U+0085, asserts a delta really carried it, and executes
the old parse as an inline negative control.

Not changed: `ensure_ascii=False` in the writer. It is correct — escaping would
inflate every non-ASCII reply, and the 27B's Chinese output is the common case.

## Rule

**A count over a filtered collection cannot detect a change that moves an element
out of the filter.** The cut and the filter shared a predicate here, so the totals
were equal by construction and the guard could not fail for the reason it existed
to check. When a control compares sizes, ask what the size is taken over.

Second: **a rerun says a failure is intermittent, never why.** The mechanism cost
two probes — one for which separators survive `json.dumps`, one for whether the
tokenizer can emit them — and both were needed, because "the separator breaks JSON"
and "this tokenizer can produce the separator" are two claims and only the pair
explains the failure.

## Results

Dev-only: four test/probe parse sites and one new test. No runtime change, so no
bench entry.

| date | commit | measurement | value |
|---|---|---|---|
| 2026-09-06 | (this) | separators `splitlines()` splits that `split("\n")` does not | **9** |
| 2026-09-06 | (this) | of those, ones that cut an SSE payload mid-JSON | **3** — `\x85` ` ` ` ` |
| 2026-09-06 | (this) | of those, reachable from sampled ids under vocab 320 | **3/3** |
| 2026-09-06 | (this) | segment counts, `split("\n")` vs `splitlines()`, on a breaking stream | **11 vs 11 — the metric was wrong, not the stream** |
| 2026-09-06 | (this) | `tests/test_server.py` after the fix | **34 passed** |
