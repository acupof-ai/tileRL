# Prefill +3.1%: lm_head ran on 512 positions to use one

> Status: Shipped

## Context

`Model.forward` ends with `lm_head` over the whole [B,T,H] activation. A decode
row has T=1 so nothing is wasted; a prefill chunk has T=512 and the engine
reads exactly one row of the result — `logits[j, chunk-1]`, the token that
starts generation.

lm_head is `[vocab=248320, hidden=5120]`. Per 512-token chunk that is
512 x 5120 x 248320 x 2 = **1.3 TFLOP**, 4.7% of a 2048-token prefill's total,
plus a 508 MB f32 logits tensor written and discarded.

## What Worked

Slice the activation to its last position before the final norm, gated on every
row being full-length (`seq_q_lens.min() == T`). A mixed tick's decode rows end
earlier and a speculative verify needs every chain position; both fall through
to the full computation.

| row | before | after | |
|---|---:|---:|---:|
| prefill/len512 | 2043.4 | 2109.7 | **1.032x** |
| prefill/len2048 | 2027.6 | 2090.5 | **1.031x** |
| prefill/len8192 | 1969.2 | 2025.8 | **1.029x** |

`accuracy/mmlu-200` held at 81.0%, which is the check that matters here: the
change picks WHICH position's logits are returned, so a wrong index would move
the answer, not the speed.

Cumulative prefill today: **1836 -> 2109.7 tok/s, 1.15x**, from this and the
pinned KV descriptors. Both were invisible in the per-kernel GPU table.

## Rule

Ask what the caller reads before optimizing what the callee computes. The
profile shows lm_head as a legitimate GEMM at a legitimate size; nothing in it
says 511 of every 512 output rows are discarded four function calls up.
