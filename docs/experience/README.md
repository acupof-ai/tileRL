# The experience archive

263 dated entries — 127 wins, 136 errors. One measurement each, written the day it was taken, kept
whether it shipped or was rejected. This page is not the listing — it is the
~16 entries that carry findings the rest of the repo rests on. Everything else
is the archive, and `git log` or `grep` is the way into it.

If you read five, read these: [zero-centered
RMSNorm](wins/2026-08-27-zero-centered-rmsnorm.md), [the thinking
cap](wins/2026-09-04-the-thinking-cap.md), [the eval cap measured
itself](errors/2026-09-04-the-eval-cap-measured-itself.md), [mma8 is not
register-bound](errors/2026-08-29-mma8-is-register-bound.md), [green checks that
proved less than they
looked](errors/2026-09-02-green-checks-that-proved-less-than-they-looked.md).

## Kernels and precision

| Entry | What it established |
|---|---|
| [fp4 GEMV decode](wins/2026-08-24-fp4-gemv-decode.md) | At M=1 a streamed-dequant GEMV beats WGMMA padded to 16 rows — the 15/16 over-compute is the whole gap. Slice decode 10.58 → 5.45 ms/tick. |
| [native fp4 w4a8](wins/2026-08-26-native-fp4-w4a8.md) | The internal grid was `e2m1fn` (no zero) against the checkpoint's OCP `e2m1`, so every load silently re-quantized a second time. Keeping the checkpoint's own format is a loader change, not a kernel change. |
| [mma8 decode GEMM](wins/2026-08-28-mma8-decode-gemm.md) | For 2 ≤ M ≤ 8 the register file, not bandwidth, is the wall: the tensor cores take the activation rows as a fragment and the twiddle already emits the B-fragment format. |
| [mma8 is register-bound — retracted](errors/2026-08-29-mma8-is-register-bound.md) | The title's own claim was wrong. mma8 issues 1.93× the load instructions for identical DRAM traffic; three fixes built on the register story all failed. A correlation is not a diagnosis. |
| [split-KV decode attention](wins/2026-08-28-split-kv-decode-attention.md) | Decode attention needs parallelism along KV length, not query heads. 32 blocks on 132 SMs was the 32k cliff — B=1 had fallen 87.5 → 28.7 tok/s. |
| [zero-centered RMSNorm](wins/2026-08-27-zero-centered-rmsnorm.md) | The 27B's garbage-logits root cause: Qwen3.5 uses `y = x_normed·(1+w)`. Three lines. Internal parity could never find it — kernel and reference shared the bug — external bisection did. |
| [Metal is green and 28× off torch-eager](wins/2026-09-04-metal-is-green-and-28x-off-torch-eager.md) | Third target to execute the one kernel tree: 256 passed, 9 skipped. |

## Engine, KV and memory

| Entry | What it established |
|---|---|
| [decode graph capture](wins/2026-08-24-decode-graph-capture.md) | Dispatch 18.3 ms → 0.04 ms. The decode tick is static, so it is replayed, not interpreted — eager dispatch alone exceeded the whole latency target. |
| [mixed-batch scheduler](wins/2026-08-25-engine-scheduler-batch.md) | Continuous batching with chunked prefill, and a fixed-width block table: the variable width was recompiling TileLang every tick. |
| [prefix-snapshot OOM](errors/2026-08-31-prefix-snapshot-oom.md) | A capacity that counts entries is not a bound when each entry owns a 150 MiB tensor. 4096 entries = 576 GiB, so eviction never fired. |
| [serve never sized its KV pool](errors/2026-09-02-serve-never-sized-its-kv-pool.md) | Every number on the card came from scripts that build the engine directly; `serve` itself asked for 275 GB of KV and held the embedding table in two dtypes. A default is only tested on the paths that reach it. |
| [prefill's chunk loop was quadratic](wins/2026-09-02-prefill-chunk-loop-was-quadratic.md) | Prefill is compute-bound where decode is bandwidth-bound, so the byte roofline that framed the task was the wrong denominator. |
| [expandable_segments is load-bearing](errors/2026-09-03-expandable-segments-is-load-bearing.md) | A published B=8 number needed an env var the tree sets nowhere. An env var that changes a result is part of the result. |

## RL and training

| Entry | What it established |
|---|---|
| [the thinking cap](wins/2026-09-04-the-thinking-cap.md) | The headline result. Train under a 256-token cap, score correctness only: GSM8K 89.6% → 94.8% uncapped (p=0.002) with 22.8% fewer tokens, and it transfers to MMLU, ARC-Easy and PIQA. A tight budget is a lever, not just a constraint. |
| [the eval cap measured itself](errors/2026-09-04-the-eval-cap-measured-itself.md) | The 39% → 94.2% that preceded it was worthless: the control saturated its own instrument. This is why the entry above exists in the form it does. |
| [GRPO existed only as a comment](wins/2026-08-29-grpo.md) | `opd_loop` was behaviour cloning with no reward anywhere. Check the capability exists before optimizing it. |
| [a GRPO group of 8 fits the card](wins/2026-09-03-grpo-27b-fits-the-card.md) | The training peak on this model is stored layer activations, not the logits — and the group mean *is* the GRPO baseline, so shrinking the group was not available. |
| [the default lr flattens the reward](errors/2026-09-03-grpo-default-lr-flattens-the-reward.md) | A sweep over a parameter has to include the value that ships. |
| [tied groups are the reward's shape](errors/2026-09-03-tied-groups-are-the-rewards-shape.md) | A tied group blames the reward and the completion length, not the sampling nucleus. Zero advantage, zero gradient. |
| [ISO optimizer](wins/2026-09-02-iso-optimizer.md) · [ISO merger](wins/2026-09-02-iso-merger.md) | Fixed-spectrum updates on the tape, and merging two specialists from their checkpoints alone — no rollouts, no data — beating the average. |
| [DFlash2 on the engine tick](wins/2026-09-03-dflash2-on-the-engine-tick.md) | Acceptance is not throughput: 6.18 of 8 accepted and 5.5× fewer trunk forwards, still 1.67× slower on the clock. Not a default flip. |

## Measurement method

The recurring failure in this repo is not a wrong kernel, it is a number that
measured the instrument. These entries are the ones that generalize.

| Entry | What it established |
|---|---|
| [green checks that proved less than they looked](errors/2026-09-02-green-checks-that-proved-less-than-they-looked.md) | Three passing checks in one day, all guarding nothing. Watch a guard refuse before trusting it. |
| ["kernel-count-bound" refuted](errors/2026-08-27-kernel-count-verdict-refuted.md) | Yesterday's top lever, acted on, measured, wrong. Price a kernel inside the graph where it runs. |
| [differencing attributed the trunk to the draft](errors/2026-09-02-differencing-attributed-the-trunk-to-the-draft.md) | 955 launches charged to the draft; its real share was 8. A cost obtained by subtracting two configurations is not a measurement of either. |
| [parity gates modelled the wrong kernel](errors/2026-08-29-parity-gates-modelled-the-wrong-kernel.md) | The project's own correct-inference gate was red for weeks behind a `tail -5`. A red parity gate is an absent one. |
| [the served prompt did not match the template](errors/2026-09-03-served-prompt-did-not-match-the-checkpoint-template.md) | The rendered chat state did not exist in the checkpoint's template. A template is executable — execute it, do not read it. |
| [depth 4 stalls are compiles](wins/2026-09-04-depth-4-stalls-are-compiles-and-block-parallel-closes.md) | Corrects an earlier entry of mine: 3 prompt groups could not see the region where the effect lives. |

## The convention

`wins/` is a change that shipped, `errors/` is one that was rejected or a
defect that was found — both are measurements, and the split is the verdict,
not the quality. Rejections are kept deliberately: most of the value above is
in knowing which levers were tried and priced out. An entry is dated, names its
arch, and carries a **Rule** — the one line the next agent may take as given
without re-deriving it. `grep -A2 '^## Rule' -r wins errors` reads every one of
them in a page.

New entries use [`wins/TEMPLATE-bench.md`](wins/TEMPLATE-bench.md), and every
runtime change needs one (`AGENTS.md`). Entries are never overwritten; a later
measurement that supersedes an earlier one links back to it and says so.

Cross-cutting write-ups — the sglang and Arle comparisons, the defect audit,
the method record behind decode 52.6 → 90.9 — are one level up in
[`docs/analysis/`](../analysis/). [`LOG-v100-sm70.md`](LOG-v100-sm70.md) is a
running work log for the V100 bring-up, numbers only.
