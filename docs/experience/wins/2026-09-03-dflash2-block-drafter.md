# DFlash2 block drafter — H20 GPU 2, 2026-09-03

> Status: Shipped (drafter + loader; engine wiring waits on the width-W verify tick)

## Context

The NextN chain drafts one token per forward, so depth d costs d draft passes.
DFlash2 drafts a whole block of 8 in one pass: the draft never runs a layer over
the prompt (a context position's K/V is a projection of the trunk's hidden state
there), and the 8 query slots — the verified anchor plus 7 mask tokens — attend
bidirectionally, with a two-tap grouped dynamic depthwise convolution around each
attention and MLP carrying order information between them.

Measured metric: acceptance length, the trunk's own greedy continuation against
one block draft. 5 short chat prompts, `Qwen3.8-27B-NVFP4` trunk +
`z-lab/Qwen3.8-27B-DFlash2`, greedy, one draft per prompt.

## What Worked

The port drafts, and every decision that could have been silently wrong has a
control that collapses without it.

| arm | mean acceptance of 8 | per prompt |
|---|---:|---|
| shipped | **5.80** | 4, 8, 7, 6, 4 |
| `target_layer_ids` read 1-based | 5.40 | 4, 8, 8, 3, 4 |
| zero-centered `+1` norm fold applied | **1.00** | 1, 1, 1, 1, 1 |
| conv's second tap zeroed | 5.20 | 4, 8, 8, 2, 4 |

The published card reports 4.39–5.46 on GSM8K/MATH/HumanEval/MBPP/MT-Bench, so
5.80 on five easy prompts is the right neighbourhood and not a broken drafter.

**The norm fold is the load-bearing decision.** DFlash2 carries `hidden_norm`
like a DSpark head, so PR #1's "DSpark ⇒ no fold" rule already covers it, but the
evidence is direct: every norm weight in the checkpoint is centred on its own
multiplier (`norm.weight` mean +2.65, `hidden_norm` +0.82, `layers.0.input_layernorm`
+0.53), while the trunk's zero-centred norms sit at −0.03. vLLM and sglang both
build every DFlash norm from their stock `RMSNorm`. Applying the fold anyway
takes acceptance from 5.80 to 1.00 — not one drafted token survives.

`read_head_params` now keys the fold on `pre_fc_norm_hidden`, the tensor only a
Qwen NextN head carries, instead of enumerating the formats that must not be
folded. A fourth format then defaults to no-fold, which is the safe way round: a
missing fold is loud, a spurious one is silent.

The walk needs only one row of the `[K, K]` transition score — its predecessor is
always the token it just emitted — so the drafter never builds the square, at
`O(steps · K · rank)` instead of `O(steps · K² · rank)`. The full square and a
sequential walk over it are the parity reference.

Parity of the whole block forward against a torch-eager transcription of vLLM's
`qwen3_dflash2.py` on a tiny random head: max abs 1.19e-07, max rel 1.07e-04
(gate rtol 1e-2). Four mutations — conv sides swapped, causal block attention,
sliding window removed, conv tap wrapped into slot 0 — each move it past 7.9e-02.

## Rule

A DFlash-family draft head's norms are plain `w·x`; the zero-centred `+1` fold is
Qwen NextN's alone, and applying it costs every accepted token, not a few. Decide
the fold from the tensor that demands it, never from a list of the ones that don't.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-03 | spec/dflash2 | H20 GPU 2 | cuda | qwen38-27b + DFlash2 | — | — | — |

Timing is pending-remote: the drafter is not on the engine's tick yet, and the
tick it will hang off (width-W graph-captured verify) is landing in parallel.
Acceptance is the number this entry claims.

Reproduce: `CUDA_VISIBLE_DEVICES=2 python scripts/probe_dflash2_acceptance.py`.
