# The page's live rate is 50.0 tok/s, and the 39-40 I reported was a口径 my own CHANGELOG had already rejected — V100 sm70, 2026-09-04

**Date:** 2026-09-04
**Arch:** cuda sm70 (V100 32GB), 27B NVFP4 + draft head, `--depth 1 --max-batch 1 --max-ctx 4096`
**Task:** #75 follow-up (the chat page's live decode-only rate)
**Instrument:** `scripts/probe_page_rate.py`, run on the pod so RTT is outside both windows
**Verdict:** The served decode rate is **50.0 tok/s** decode-only and **46.3 tok/s** wall.
Both are right; they answer different questions. The **39.3-40.2** I reported earlier the
same day is neither — it is `wall_ms / tokens`, which this repo's CHANGELOG had already
recorded as a broken instrument in the entry directly above it.

## Context

The chat page now shows a live rate, windowed from the first token so prefill is excluded.
Verifying it against the pod produced 50.3 tok/s — **1.257x** the 39.3/40.0/40.2 I had
reported hours earlier off the same server. A 1.257x jump with no engine change is a
measurement problem, not a win, so it needed attribution before any number was quoted.

## What worked

**Three candidates, eliminated in order by measurement.**

**1. Acceptance moved 1.010x, not 1.257x.** From `/health` after five requests:
`tokens_generated 1000 / decode_forwards 545` = **1.835 tok/forward**, `spec_accepted 452 /
spec_drafted 545` = **acceptance 0.829**, against the 1.795 / 0.821 recorded earlier. So
`enable_thinking` changing what the draft predicts — the obvious suspect, since the page now
sends it — accounts for **1.022x of the 1.257x**. Not the cause.

**2. The口径 gap is 1.081x.** Printing both windows for the same request, in the same run:

| run | ttft | tokens | decode | wall |
|---|---:|---:|---:|---:|
| 0 | 366 ms | 200 | 50.3 | 46.0 |
| 1-4 | 324 ms | 200 | 50.0-50.2 | 46.2-46.5 |
| median | | | **50.0** | **46.3** |

`decode / wall = 1.081x`, which is the prefill and the queueing. Real, and still not 1.257x.

**3. Generated length is not it.** Sweeping `max_tokens`, the model stops itself at 390:

| max_tokens | got | decode |
|---:|---:|---:|
| 200 | 200 | 50.3 |
| 400 | 390 | 49.5 |
| 800 | 390 | 49.4 |
| 1600 | 390 | 49.4 |

200 → 390 tokens costs **1.8%**. Context is a known strong variable (#47/#57) but not over
this range.

**So the gap is in the earlier measurement, and the repo already said so.** CHANGELOG's
first-visit entry, published the same day, lists exactly this among three failed
instruments: *"`wall_ms / tokens` charges prefill to decode (#24's 31 ms/prompt token is
~310 ms of a 1650 ms request, against a bench figure that is decode-only by construction)
and read as a **15% serve regression that does not exist**"*. Its own correct figures are
**46.1 / 42.0 / 44.3 / 45.0** — and this run's wall number, **46.3**, matches the 46.1 to
**1.004x**.

Three numbers, one server, no contradiction:

| number | window | status |
|---|---|---|
| 39.3-40.2 | `wall_ms / tokens` | rejected instrument, per this repo's own entry |
| 46.3 | request sent → last frame | matches the recorded 46.1 (1.004x) |
| 50.0 | first frame → last frame | the page's number, decode-only |

## Rule

**A rate without its window is not a number.** Print the divisor, not just the quotient:
`probe_page_rate.py` emits decode and wall side by side from one request so the 1.081x
between them is visible rather than reconstructed a day later.

And the sharper one, because the answer was already written down: **before attributing a
gap to the system, grep the CHANGELOG for the instrument.** I spent three measurements
eliminating candidates for a discrepancy whose cause was named, in prose, in the entry
above mine — including the phrase "a 15% serve regression that does not exist", which is
the same mistake in the same direction. Reading it first would have cost one grep instead
of five pod runs.
