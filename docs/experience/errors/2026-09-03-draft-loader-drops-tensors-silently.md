---
question: Could the draft-head loader's silently dropped tensors be the source of the all-NaN verify logits?
status: measured
source: tileRL, branched from aafe162 (spec/dflash2); investigated 2026-09-03
---

# The loader does drop tensors silently, and it is not the NaN source

Two questions, asked in order. The first found a real defect. The second refuted
the hypothesis it was raised to support.

## 1. Does `read_head_params` silently drop tensors? Yes — 11 of them

`spec.py:read_head_params` classifies each checkpoint tensor three ways and only
two of them are visible:

```
key = stems.get(tail)                  # the head's own tensors -> loaded
mapped = _param_key_for(bare)
if mapped in (embed_tokens, lm_head, final_norm): skipped.append(...)   # warned
elif mapped is not None:               params[mapped] = ...             # loaded
                                       # mapped is None -> falls off the end
```

The third case has no `else`. A tensor `_param_key_for` does not recognise
vanishes with no warning and no record.

Measured by running the real loader over a safetensors carrying the published
DFlash2 tensor names (2 layers, 36 tensors):

| reader | loaded | warned | **silently dropped** |
|---|---|---|---|
| `_DFLASH2_TOP` (correct for this checkpoint) | 36/36 | 0 | **0** |
| `_DRAFT_TOP` (Qwen NextN / DSpark) | 25/36 | 0 | **11** |

The 11 are exactly the three families in the bug report:
`candidate_selector.{hidden_projection,predecessor_codebook,successor_codebook}`
and, per layer, `attention_conv.{base_kernel,kernel_projection}` and
`mlp_conv.{base_kernel,kernel_projection}`.

The published 5-layer checkpoint at `/work/Qwen3.8-27B-DFlash2` gives the same
answer at its real size — **81 of 81 through `_DFLASH2_TOP`, 0 unknown**, and 23
of 81 unknown through `_DRAFT_TOP`. The count scales with layer depth (4 conv
tensors per layer plus the 3 selector ones); 11 above is the 2-layer figure. The
raise therefore cannot fire on the path this checkpoint actually takes.

So the report's "`attention_conv`, `mlp_conv` and `candidate_selector` map to
`None` and are silently dropped" is accurate — **for a DFlash2 checkpoint read
through the NextN map.** `load_draft` sniffs `candidate_selector.` and routes to
`load_dflash2`, so the ordinary path is correct; any caller reaching
`read_head_params(path, _DRAFT_TOP)` directly gets the crippled head.

## 2. Can a dropped weight produce NaN? No — it raises

This is what decides whether the defect explains the 372/560 all-NaN verify
ticks. It does not.

Every conv and selector weight is read by **direct subscript**. Counted
mechanically over `DFlash2Head`: **19 `self.params[...]` subscripts, zero
`self.params.get(...)`**. `_conv_in` (`dflash2.py:176`) does
`self.params[f"{conv_key}.proj"]`, so a dropped tensor is a `KeyError` on the
first drafted block:

```
        complete: (runs)
conv+selector dropped: KeyError: 'layers.0.attn_conv.proj'
```

A crashing loader cannot emit token id 0 five hundred times. **The NaN source is
elsewhere** — it stays with the split-KV guard and the verify tick, which other
work owns.

This mattered to establish rather than assume: had one of those 19 lookups been
`.get()`, the head would have drafted from garbage instead of crashing, every
DFlash2 acceptance number on record would have been suspect, and the fix would
have been a different one. The tolerance of the *reader* decided which bug this
was.

## The head the bench actually used loads completely

`/work/Qwen3.8-27B-NVFP4/model_mtp.safetensors` — the head behind every
acceptance number on record — was checked because a silent drop on its path would
have put those numbers in doubt. It does not have one.

15 tensors, no `candidate_selector.*`, so `load_draft` routes it to `_DRAFT_TOP`,
which is the correct reader for it: it is a Qwen NextN head, carrying both
`pre_fc_norm_embedding` and `pre_fc_norm_hidden`.

| reader | loaded | warned | silently dropped |
|---|---|---|---|
| `_DRAFT_TOP` (the path it takes) | **15/15** | 0 | **0** |
| `_DFLASH2_TOP` (for contrast) | 13/15 | 0 | 2 (both pre-fc norms) |

`load_draft`'s required set `{fc, norm, pre_fc_norm_hidden}` is complete. **So the
DFlash2 acceptance record stands** — the silent drop never touched the bench path,
and the new raise does not break it.

The contrast row is the reverse mismatch, and the new raise now catches it too: a
NextN head read through the DFlash2 map would have lost both pre-fc norms
silently, and the `+1` fold is keyed on `pre_fc_norm_hidden` being present, so it
would also have skipped the fold. That is the failure the entry above records
finding once already — a missing fold ranked the head's argmax 248191 of 248320.

## Fix

`read_head_params` raises on the third class, naming the count and the first
eight offenders. A tensor the map does not name means the wrong reader for this
format, or a key this port does not implement — both are worth stopping for, and
neither is dead weight.

## What the guard was watched failing

Removing the raise makes `test_unmapped_tensors_raise_instead_of_vanishing` fail
with `DID NOT RAISE RuntimeError`. The test also carries its own negative control
in the same body: the correct reader (`_DFLASH2_TOP`) loads the same file with
`attn_conv.proj` and both codebooks present, so "it raised" is distinguishable
from "this file is unreadable".

## Rule

**A three-way classification needs three visible branches.** Two named cases and
a fall-through is not a classification; it is two cases and a leak. When reading
an external format, the unrecognised case is the one that must be loudest —
it is the only one that means "you are not reading what you think you are".

And: before fixing a defect found by reading, measure which *kind* of failure it
produces. "Silently dropped weights" and "NaN logits" sound like cause and
effect and here they are unrelated — a strict reader turns the first into a crash
that can never reach the second.

## Scope

Measured on `main` (`dflash2.py`, `spec.py`, `tests/test_dflash2.py` and
`probe_dflash2_acceptance.py` all live there). The 27B checkpoint contributed key
names only — no weights were loaded and no card was used, because the routing and
coverage questions are answered by the names.

**A wrong claim this entry originally carried**, kept because the mistake is the
reusable part: it said `dflash2.py` was not on `main`. It is. The grep that
"proved" otherwise ran in a shared checkout whose working tree was pinned at
`a702c9a` from earlier in the session and had never been fetched. "My tree does
not have it" was reported as "main does not have it" — the exact substitution that
makes a stale-tree observation look like a claim about the branch. `git ls-tree
origin/main` answers the question the working tree cannot.
