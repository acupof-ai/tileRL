# GRPO completion-length buckets — cpu, 2026-09-06

> Status: pending-remote (27B performance)

## Context

The [run-2 record](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md)
reports a 126.9 s median backward at a 2048-token cap. The training rectangle
used the cap even when completions were short.

## What Worked

Round the group's longest completion up to a power of two, with a 256-token
floor (or the rounded cap for caps below 256), **clamped to the cap**. At cap
2048 the completion widths are 256, 512, 1024 and 2048; prompt width is added
unchanged. `seq_lens` and the advantage mask are unchanged.

**The clamp is not cosmetic.** Without it a cap that is not a power of two
rounds PAST itself: cap 1500 with a 1400-token longest completion gives 2048, a
548-token overshoot of padding the mask discards, and cap 300 gives 512. No
recipe on main has such a cap — every `max_new_tokens` is 8, 256 or 2048 — so
this was latent rather than broken, and it cannot corrupt anything either way:
`rl_step` takes no engine, so the padded rectangle reaches the tape and never
the paged KV pool. The cost was memory and backward FLOPs.

The clamp adds no JIT. Checked exhaustively over every `(cap, longest)` pair to
2099: **zero cases where the width exceeds the cap or falls below the longest
completion**, and the width set is the powers of two from 256 up to the cap plus
the cap itself — `{256, 512, 1024, 2048}` at cap 2048, `{256, 512, 1024, 1200}`
at cap 1200, `{256, 300}` at cap 300, `{8}` at cap 8. Same size as the unclamped
expression's at every cap to 2099, with only the largest entry differing (cap
instead of the power of two above it), so the clamp trades no extra JIT for the
saved padding.
`min(bucket, cap)` was the right shape; capping by rounding the cap
down would have reintroduced the data-dependent widths the buckets remove.

The CPU tiny gate runs three arms: longest 300 and 1100 at cap 2048 (widths
prompt+512, prompt+2048), and **longest 1200 at cap 1200**, which is the arm
that sees the clamp — at 2048 the round-up lands on the cap itself and the
defect is invisible. Restoring the unclamped expression fails that third arm
with `(2, 2051) == (2, 1203)` while both 2048 arms still pass. It compares
real-completion cross-entropy
and pre-clip parameter gradients with the old 2048-wide rectangle, at the same
seed, with fp32 tolerances rtol=1e-5, atol=1e-6. The returned diagnostic CE
still includes padding and is not the real-token loss compared by this gate.
The gate passes; restoring the pre-bucket width expression fails with
`AssertionError: wrong completion bucket width`: (2, 2051) versus (2, 515).

The step line now prints `width` beside `tok`, because run 2 could not tell a
long tail from every completion sitting at the cap without both numbers.

Each new bucket pays one shape JIT for a fixed prompt width and group size.
The [original width measurement](2026-09-03-grpo-step-width-is-a-jit-key.md)
and `grpo_loop` comment report tiny's 37.7 s for a new width versus 71 ms for a
repeat. This is historical evidence, not a new timing or a 27B estimate.

## Rule

Bound compiled widths with buckets while keeping true lengths in the loss mask.

## Results

27B bucketed backward and whole-step timings: pending-remote; no GPU measurement
was made here. CPU correctness command:
`TILERL_TARGET=cpu uv run pytest tests/test_rl.py -k length_buckets`.
