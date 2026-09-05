# GRPO completion-length buckets — cpu, 2026-09-06

> Status: pending-remote (27B performance)

## Context

The [run-2 record](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md)
reports a 126.9 s median backward at a 2048-token cap. The training rectangle
used the cap even when completions were short.

## What Worked

Round the group's longest completion up to a power of two, with a 256-token
floor (or the rounded cap for caps below 256). At cap 2048 the completion
widths are 256, 512, 1024 and 2048; prompt width is added unchanged.
`seq_lens` and the advantage mask are unchanged.

The CPU tiny gate forces longest completions of 300 and 1100 tokens and checks
widths prompt+512 and prompt+2048. It compares real-completion cross-entropy
and pre-clip parameter gradients with the old 2048-wide rectangle, at the same
seed, with fp32 tolerances rtol=1e-5, atol=1e-6. The returned diagnostic CE
still includes padding and is not the real-token loss compared by this gate.
The gate passes; restoring the old width expression fails with
`AssertionError: wrong completion bucket width`: (2, 2051) versus (2, 515).

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
