# The dequantized bf16 checkpoint the sglang comparison ran on emits garbage — 2026-08-28

## Context

`docs/experience/2026-08-28-vs-sglang-h20.md` compares tileRL against sglang on
the same card and the same model. sglang has no NVFP4 path on Hopper, so the
comparison runs it on `/work/Qwen3.8-27B-bf16`, produced by
`scripts/dequant_to_bf16.py` from the NVFP4 checkpoint. That comparison measured
**throughput only** — nobody ever read a token it produced.

MMLU did. Unconstrained, one greedy token, n=1000:

| engine | model | MMLU 0-shot | unparsed | sample completions |
|---|---|---:|---:|---|
| tileRL | Qwen3.8-27B-NVFP4 | 76.3% | 0 | letters |
| sglang | /work/Qwen3.8-27B-bf16 | **0.0%** | 998 | `'Fd'`, `' Ad'`, `'束'`, `'tz'`, `'炉'` |

An earlier grammar-constrained run of the same checkpoint scored 20.7% (chance
is 25%) with a broken byte before each letter — which looked like an FSM bug and
masked the real one for an hour.

## Root Cause

Not yet isolated — the confirming single-prompt generation was queued and died
when another tenant took 72 GB of GPU 7. What is established: the bf16
checkpoint produces incoherent text on the same prompts where the fp4 original
produces 76.3% MMLU, so the defect is in the dequantized copy, not in sglang and
not in the model. Candidates, in order: the ModelOpt global-scale convention in
`dequant_nvfp4(..., global_divide=True)`; the fp4 branch never casting to bf16
(the fp8 branch does); a missing tensor (the source has `model.safetensors` plus
`model_mtp.safetensors`, and the writer reads only the first).

## Fix

Pending. Until it lands, every sglang number is throughput-only and carries this
caveat; no accuracy comparison exists.

## Rule

A perf comparison against another engine is not valid until you have read its
output. Same shapes at the same speed says nothing about whether the weights
survived the conversion — garbage decodes just as fast. Score both engines on
one accuracy task before publishing a single throughput number.
