# tileRL

**Serve and RL-train Qwen3.8-27B (NVFP4) on one Hopper card, in one process.**

sglang refuses this checkpoint on Hopper. tileRL runs it in its own TileLang
kernels. A served agent turn is 94% prefill — 294 of 314.33 s
([wins/2026-09-07](docs/experience/wins/2026-09-07-a-claude-code-turn-is-314-seconds-of-prefill.md)) —
so the table leads with prefill, not decode:

| one H20, Qwen3.8-27B | workload | decode tok/s | prefill tok/s |
|---|---|---:|---:|
| **tileRL**, NVFP4 + FP8 | d512, 64 out | 92.4 | **2689.8** |
| sglang, bf16 (cannot load NVFP4 on Hopper) | d512, 64 out | 54.2 | 2512 |
| sglang, online fp8 | d512, 64 out | 39.9 | **4022** |
| **tileRL**, W=8 block speculation | 200 GSM8K, 512 out | **126.5** | — |
| **tileRL**, same arm, speculation off | 200 GSM8K, 512 out | 79.5 | — |

On prefill (94% of the turn) tileRL beats sglang's bf16 arm (2689.8 vs 2512)
and loses to its online-fp8 arm (4022); on decode (6%) it leads 92.4 to 54.2.
B=1 decode is the rollout shape — the training target — but not the serving
bottleneck. Weights are fp4 against **bf16** activations at B=1; the fp8 path
is the M > 1 kernel and carries prefill and batched decode.

## Status (2026-09-13)

**Development stops here in favour of ecosystem engines (ckl, 2026-09-13); the V100 serve endpoint stays up at `n37-002-027:8000`.** This is the state at wrap-up.

What runs today (H20 unless noted; every number links its entry):
- **Dense serve**: the table above — 92.4 d512 decode / 2689.8 prefill tok/s, 74.6% MMLU
  ([wins/2026-09-07](docs/experience/wins/2026-09-07-a-claude-code-turn-is-314-seconds-of-prefill.md)).
- **Sparse KV is accuracy-equivalent but slower**: on the same matched 400 thinking MMLU
  questions, `--sparse-k 128` scores **0.863 vs dense 0.858** at **37.4 vs 187 tok/s**,
  so it stays opt-in on perf grounds
  ([errors/2026-09-12-sm90-fused-attn-prep-sparse-packed-prefill.md](docs/experience/errors/2026-09-12-sm90-fused-attn-prep-sparse-packed-prefill.md)).
  Sparse decode is graph-captured once opted in on sm90 (pure sparse: `--sparse-k` without `--sparse-min-tokens`); hybrid sparse ticks and every sm70 sparse tick run eager
  ([#557](https://github.com/acupof-ai/tileRL/pull/557)); sparse B=1 eager at 32k is
  **88.23 ms/tick vs ~13 ms dense graph**, 256k pending
  ([#571](https://github.com/acupof-ai/tileRL/pull/571)).
- **Speculative decode is a B=1 lever only**: B=8 free-running spec is not reproducible
  across identical cold waves (3/8 rows agree) and is **not** verified — sparse+spec
  warm prefix adoption is exact at B=1
  ([errors/2026-09-13-sm90-b8-spec-wave-not-reproducible.md](docs/experience/errors/2026-09-13-sm90-b8-spec-wave-not-reproducible.md),
  [wins/2026-09-13-warm-spec-prefix-adoption.md](docs/experience/wins/2026-09-13-warm-spec-prefix-adoption.md)).
- **V100 sm70 runs a stable dense serve** — 4 slots / 8192 ctx, verified 2026-09-13
  (4×7.4k concurrent prefill peak 27.2/31.7 GiB, 0 OOM; soak 234 turns 0 errors).
  Endpoint, config and operations: [docs/serve-v100.md](docs/serve-v100.md).
  Sparse is limited there: 64k serves at B=1 (3.44 tok/s decode, 343.2 s prefill,
  RSS 27.92/31 GiB) and the
  **256k prefill is SIGKILLed (OOM)** before its spill file is ever written
  ([errors/2026-09-13-v100-256k-sparse-prefill-host-oom](docs/experience/errors/2026-09-13-v100-256k-sparse-prefill-host-oom.md)). P6 fp8 long-ctx fits are ledger-derived,
  not card-measured ([wins/2026-09-11-p6-long-context-budget-on-one-h20.md](docs/experience/wins/2026-09-11-p6-long-context-budget-on-one-h20.md)).
- **RL**: GRPO 100-step tiny gains reproduced on GSM8K (89.6→94.8%) but not MATH-5; the
  full-27B training rollout tick is measured **2.64x the serving tick** and never closed
  ([errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md](docs/experience/errors/2026-09-08-the-training-rollout-tick-is-2.6x-serving.md)).

Unfinished (owners in [OPEN.md](docs/experience/OPEN.md), **17 open defects**): the sparse 256k
decode number is frozen with the other H20/sm90 rows (H20 stopped 2026-09-16; not awaiting a lend) — its 32k row shipped in
[#571](https://github.com/acupof-ai/tileRL/pull/571) (88.23 ms/tick vs ~13 ms dense graph); 32k
teacher-forced sparse NLL closed partial at **5 of 8 windows** (k128 gap −0.1402 nats/token,
[entry](docs/experience/errors/2026-09-13-sparse-nll-32k-partial-5of8.md)).
All eight H20 cards were recorded as transferred to aupai on 2026-09-13; as of 2026-09-14
H20-dependent rows are deferred by decision (not awaiting a lend), and the V100 sm70 endpoint
stays the live verification platform.

Accuracy is not in that table: these weights score **74.6% MMLU 0-shot**
(746/1000, `fuse_projections=True` via `scripts/mmlu.py`, 2026-09-03; the
unfused arm scores 74.2% — [why](docs/experience/errors/2026-09-03-mmlu-score-depends-on-concurrency.md)).
Both sglang arms run a dequantized bf16 checkpoint that emits garbage, so the
rows above compare kernels, not accuracy.

**Read the workload column before comparing rows.** Only the first three are
the same shape; the speculation pair ran 200 real GSM8K problems, so 126.5 is
read against its own 79.5 base (**1.591x**, derived), never against 54.2.
Speculation is a B=1 lever: at B=8 it lands at 0.928x. Speculation and a real
prefix cache cannot both be on when the drafter taps the trunk's aux layers
(the DFlash2 block drafter): an adopted prefix skips the positions the
draft's context was built from, so the engine rejects the combination at build.

This table read **135.5** for the speculation row until 2026-09-09: re-measured
on the current sha, same workload, a different H20, it reads **126.5** warm —
base and drafter reproduce, throughput did not. 135.5 stands in its dated
entry, not here ([arm](docs/experience/wins/2026-09-03-batched-selector-walk.md)).

**The same checkpoint runs on a V100** — sm70, no bf16 or fp8 path, two
generations before NVFP4: 50.0 tok/s decode-only, 19 GB of weights off disk.
Not the perf target; evidence the kernels are not bound to one arch.

```
curl http://n37-002-027:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"hi"}]}'
```

## One runtime, not two

GRPO and self-teacher distillation roll out through the same engine and the same
weights — LoRA on the frozen fp4 base, no weight sync, no second stack.
`train --recipe` → `ledger` → `merge` → `serve`, every run writing a manifest
with its inputs and its gates.

`merge --method` ships the literature baselines next to ISO — TIES (trim, sign
election, disjoint mean) and DARE (Bernoulli drop plus rescale). On the tiny
harness both keep both specialist tasks and land next to averaging (base
22.34/21.99, TIES 18.02/16.45, DARE 18.45/17.55, ISO 16.00/14.73); the 27B
ISO-vs-baselines verdict stays `pending-remote`
([entry](docs/experience/wins/2026-09-11-ties-dare-merge-baselines.md)).

The memory and kernel costs are one model, not profiled: a `precision.Format`
plus `nbytes` prices every tensor, `memory.plan` is the allocator's input, and
each kernel declares its bytes/flops against a per-card measured floor.
`serve --dry-run` and `train --dry-run` print the ledger without a card —
served 27B weights are an exact header-derived **24,436,981,888 B** and the
checkpoint-faced decode tick streams **22.36/25.63 GB at B=1/B=8** — and
`ledger --devices` shows measured calibration and residency
([design doc](docs/design-cost-model.md)).

The gated-delta layer trains under context parallelism: the reverse tape runs
cross-rank for the affine state transfer (conv kernel 1) and the conv halo
(kernel 4), with world2 gates green on CPU and the card run listed in the
[pending-remote runbook](docs/experience/PENDING-REMOTE-CARDS.md). The P6 long-
context lines are derived from the same ledger, not measured on a card yet:
**256K B=1 fits in bf16; 8×128K fits only with fp8 KV; 8×256K misses one card**
([entry](docs/experience/wins/2026-09-11-p6-long-context-budget-on-one-h20.md)).

**Sparse KV is the long-context lever; it is opt-in, under investigation, not
the default.** `--sparse-k 128` enables sparse Quest selection — top-128
16-token pages plus an 8-page causal window, one softmax over both — with pages
not selected this tick demoting through the pinned-host path and promoting back
through one batched H2D/D2H pair per tick, so long contexts hold in a small
device hot pool. No attention kernel changed: the sparse tick is a packed block
table. `tilerl serve` defaults to dense (`DEFAULT_SPARSE_K = 0`).

Sparse shipped briefly as the default (#530) and was reverted the same day by
[#558](https://github.com/acupof-ai/tileRL/pull/558) after end-to-end MMLU on
the 27B H20 exposed a generation defect the short-context V100 continuity probe
did not: on the same 400 thinking questions, greedy spec-off sparse k=128
scored **0.3475 accuracy vs dense 0.9150**; the spec-on run was worse (paired
n=1282 acc 0.202 vs dense 0.859, McNemar delta −0.624 ±0.045), with sparse
generations ballooning to ~600–865 tokens and often emitting no answer letter.
At MMLU lengths k=128 selects the full page set, so this is not expected
near-tie divergence from a smaller attention set — it is a defect on the
sm90 B=8 sparse generation path under localization. The earlier V100 rows that
motivated the default still hold for what they measured: after the #546
own-table `page_base` fix, k=all is token-identical on sm70, and 8k/32k
prefill KL/top-1 (0.0023/0.981, 0.019/0.949) plus a 64-token teacher-forced
NLL gap (~+0.013 nats/token, n=3) are short, forced-teacher, B=1 sm70 numbers;
they do not predict long free-running B=8 sm90 generation
([entry](docs/experience/wins/2026-09-12-dense-vs-sparse-long-ctx.md),
[#531](https://github.com/acupof-ai/tileRL/pull/531),
[#558](https://github.com/acupof-ai/tileRL/pull/558)). Do not re-enable sparse
by default until an sm90 B=8 long-generation continuity gate passes.

**The self-judge retry recipe (P1) is rejected**: across two matched seeds the
held-out GSM8K gain has opposite signs (458 → 448 and 456 → 482), so it does not
ship on the winning seed
([verdict](docs/experience/errors/2026-09-11-p1-judge-recipe-two-seed-rejected.md)).

**The thinking cap buys economy. Whether it buys accuracy is unsettled.**
Cap the rollout at 256 tokens, score correctness only, then measure uncapped — the
policy finds the shorter path to the same answer.

| GSM8K, uncapped, n=500 — measured 2026-09-04, one run | accuracy | total tokens | tokens / correct |
|---|---:|---:|---:|
| base | 89.6% | 157,601 | 351.8 |
| after 100 GRPO steps | **94.8%** | **121,642** | **256.6** |
| | +5.2 pts, p=0.002 | **−22.8%** | **−27.1%** |

**These rows have not been reproduced since.** A MATH level-5 run reproduced
the token cut and not the accuracy (**−27% tokens, −5 points**, McNemar
p=0.18), so the two jointly support the economy claim, not the accuracy claim
in either direction.

It transfers to tasks the adapter never saw — tokens fall 22.0% on MMLU, 18.9%
on ARC-Easy, 22.9% on PIQA, with no measurable accuracy change at n=100. Output
tokens are the serving bill, so this is a win in the units the product is sold in.

The control moved the claim. Retrained at 2048, the policy solves 96.6% of
training prompts and **92% of GRPO steps carry no gradient** — the tight cap
held the task hard enough to keep groups mixed — yet still reaches **96.4% on
GSM8K off 8 gradient steps**: the cap demonstrably buys sample efficiency and
the token cut, not the accuracy. Ranking two arms 1.6 points apart needs ~2,600
questions each and these runs were 500.

[The result](docs/experience/wins/2026-09-04-the-thinking-cap.md) ·
[the control that reinterpreted it](docs/experience/wins/2026-09-04-the-cap-was-the-gradient.md) ·
[why the first number was wrong](docs/experience/errors/2026-09-04-the-eval-cap-measured-itself.md)

On MATH level 5 a correctness-only reward lengthens rollouts to the cap and
ties every group — run 2 was killed at step 45 of 100. The token cut belongs
to a tight cap on a solved task, not the reward.
[The run](docs/experience/errors/2026-09-06-the-rollouts-grew-into-the-cap.md)

## Quickstart

```bash
uv sync
TILERL_QWEN38_SOURCE=/path/to/Qwen3.8-27B-NVFP4 uv run tilerl serve --model qwen38-27b
```

OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` — point a client, or Claude
Code, at it. No GPU: `uv run tilerl serve` runs the tiny model on CPU.

```bash
uv add datasets   # only for the GSM8K dump below; not a runtime dependency
python scripts/gsm8k_jsonl.py train gsm8k.jsonl
uv run tilerl train --recipe grpo-gsm8k-27b --data gsm8k.jsonl --eval-gsm8k gsm8k_test.jsonl
uv run tilerl ledger
```

## Where things are

[`docs/roadmap.md`](docs/roadmap.md) — phases and gates ·
[`docs/design-cost-model.md`](docs/design-cost-model.md) — the one byte/kernel model and its ledger ·
[`docs/design-rl-stack.md`](docs/design-rl-stack.md) — ISO, the draft head, the ledger ·
[`docs/design-engine.md`](docs/design-engine.md) ·
[`docs/design-kernels.md`](docs/design-kernels.md) ·
[`docs/support-matrix.md`](docs/support-matrix.md) — per-op status per target ·
[`docs/experience/PENDING-REMOTE-CARDS.md`](docs/experience/PENDING-REMOTE-CARDS.md) — the gated card work that remains ·
[`docs/experience/`](docs/experience/) — every measurement, win and dead end, dated ·
[`AGENTS.md`](AGENTS.md) — the gates a change clears

Every number above sits in a dated entry under `docs/experience/`. The prefill,
KV-reuse, training and batched-decode rows are additionally held by `bench`
against [`bench-baseline.json`](docs/experience/wins/bench-baseline.json) at
≥ 0.97×; the single-stream decode rows (92.4 among them) lost their baseline
key when five rows' commits could not be recovered — a guessed sha is forged
provenance. Measurements append to the validated store
(`docs/experience/bench/measurements.jsonl`); a metric without a store row rests
on its entry, never a datasheet number. `uv run pytest` gates every commit.
