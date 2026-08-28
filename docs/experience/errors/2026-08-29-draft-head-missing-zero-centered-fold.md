# The draft head's norms missed the zero-centered +1 — 2026-08-29

## Context

The bundled NextN/MTP draft head produced logits ANTI-correlated with the
trunk: the trunk's own argmax ranked **248191 of 248320** in the draft's
distribution (random would be ~124160), top-1 agreement 0.0%, max
probability 0.0003 against a uniform 4.03e-06. Both `fc` concatenation orders
gave the same result, so the fault was upstream of the fc.

## Root Cause

`Qwen3_5RMSNorm` is zero-centered — `y = x_normed * (1 + weight)`.
`load_hf` folds the +1 into the weight at load (`model.py:954-958`) so the
kernel stays a plain RMSNorm. `load_draft` reads the head's safetensors
directly and skipped that fold, so every norm in the head scaled by ~0.2
instead of ~1.2 (`std norm(hidden) 0.1922` in the diagnostic).

The engine was never wrong about this. `verify_lens` read the head's ~0.0003
confidences, decided speculation was not worth a verify row, and returned 0 for
every request — `spec_drafted` stayed 0 and each tick paid four wasted draft
forwards (460 ms vs a 66 ms baseline tick). The policy behaved correctly on
garbage input.

## Fix

`load_draft` applies the same fold to every norm in the head
(`input_norm`, `post_attn_norm`, `q_norm`, `k_norm`, `norm`,
`pre_fc_norm_hidden`, `pre_fc_norm_embedding`).

| | before | after |
|---|---|---|
| trunk argmax's rank in the draft's distribution | 248191 / 248320 | **0** |
| teacher-forced top-1 agreement | 0.0% | **84.4%** |
| mean max-probability | 0.0003 | 0.562 |

## Rule

Anti-correlation is a wiring signal, not a quality signal: bad weights give a
random rank, only a systematic error gives the bottom of the distribution.
Reach for the scale diagnostic before the semantic one.

And: a second loader for the same checkpoint family inherits every convention
the main loader applies. The conventions live in `load_hf` as suffix rules, not
in the file format, so nothing about the tensors themselves says they are
missing — list what the main path does to a tensor between disk and use, and
check the second path does each one.
