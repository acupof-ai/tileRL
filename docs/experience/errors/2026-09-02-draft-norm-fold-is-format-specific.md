# The draft-head norm fold was unconditional, and it is format-specific — 2026-09-02

## Context

Assessing whether the `RadixArk/Qwen3.8-27B-DSpark` head on ModelScope could be
wired into `spec.load_draft`. Reading the loader against the official
`dspark.py` / `dflash.py` (both ship in the model repo) and against
agent-infer's own DSpark implementation turned up a live bug that no
measurement here would have caught.

## Root Cause

`spec.py:200` folded `+1` into every norm unconditionally:

```python
for k, v in params.items():
    if k.endswith(("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")):
        params[k] = (v.float() + 1.0).to(v.dtype)
```

That is right for a **Qwen NextN** head, whose norms are zero-centered
`Qwen3_5RMSNorm` (`y = x_normed * (1 + w)`). It is wrong for a **DSpark** head,
whose norms are plain `w * x`. agent-infer settles it:
`infer-cuda/src/qwen35/dspark.rs:580,726` loads `hidden_norm`, `norm`,
`input_layernorm` and `post_attention_layernorm` with `load_vec_any` — a plain
kernel — and only `q_norm` / `k_norm` with `load_vec_minus_one`, because the
downstream prep kernel hardcodes `(1 + w)` there.

So a DSpark checkpoint's `1.0` became `2.0`, with no exception and no warning.
This is the mirror image of
[errors/2026-08-29](2026-08-29-draft-head-missing-zero-centered-fold.md): same silent scale
corruption, opposite direction — and this direction produces none of the
anti-correlated-logits signature (argmax ranked 248191/248320) that made the
first one findable. It would have surfaced as "the new head drafts badly",
which reads as a model-quality problem, not a loader bug.

Two smaller conventions were wrong the same way, both silent-then-loud:

- Depth was `1 + max(index)`, so an absolute-index head (DeepSeek numbers its
  MTP layer by its position in the trunk) inferred depth `index+1`, allocated
  that many draft KV planes, and died on a missing `layers.0` — pointing at the
  wrong thing.
- `_param_key_for` mapped top-level `embed_tokens` / `lm_head`, so a head
  shipping its own would load them, and `engine._quantize_draft` packs anything
  2D: 2 × 2.5 GB of dead weight at 248320×5120, on a card that has already
  OOMed at 31.3 GB. `forward` never reads either — it takes the embedding from
  the trunk (spec.py:133) and reads out through the trunk's lm_head (:150).

## Fix

Decide the flavor from the source name, which the mapping loop already sees:
`hidden_norm` ⇒ DSpark ⇒ fold nothing; `pre_fc_norm_hidden` ⇒ NextN ⇒ fold as
before. Plus: require layer indices to be exactly `0..n-1`, and skip
vocab-projection keys with one warning.

`tests/test_draft_loader.py` gates both conventions — a norm of 0.25 must load
as 1.25 in NextN and 0.25 in DSpark, and a layer numbered 7 must raise naming
the indexing.

## Rule

A checkpoint-format assumption stated in a comment is a bug waiting for the
second format. `spec.py:166-173` already claimed to accept both flavors — the
mapping table had a `hidden_norm` entry — while the code twenty lines below
assumed only one. When a loader advertises N formats, every transform in it
needs to say which format it belongs to.

Second: when a reference implementation exists, read it before inferring the
convention from the weights. agent-infer had already answered this question in
its choice of two loader functions; four hours of reasoning from `config.json`
would not have.
