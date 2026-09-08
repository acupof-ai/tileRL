# A dispatch whose default was a different model

**Date:** 2026-09-08
**Session:** v100-sm70-fp4-55

## Context

`_build_model` (`cli.py:47`) dispatches on the model NAME. `qwen38-27b` loads the
checkpoint; everything else falls through to `config.tiny()` — a random 64-hidden, 2-layer
model. The fall-through was written for `tiny` and `tiny-agent`, and it accepted every other
string as well.

Found while driving `scripts/steps_to_reward.py`'s `--model` flag before a pod run. Measured
rather than read:

| passed | built |
|---|---|
| `qwen38_27b` (underscore typo) | `tiny`, hidden 64, layers 2, vocab 320 |
| `Qwen38-27B` (capitalization) | `tiny` |
| `/data00/models/Qwen3.8-27B-NVFP4` (the real path) | `tiny` |
| `27b`, `""` | `tiny` |

Against the checkpoint's 5120 / 64 / 248320. **No exception, no warning.** A run finishes and
prints a table that reads like the 27B.

## Root cause

The dispatch's default branch was a *model*, not an error. Two things follow from that and
only the second is obvious.

The obvious one: a typo silently changes what was measured. The other is that the defect is
invisible at every call site. Six scripts pass a user-supplied `--model` straight into it with
no argparse `choices`:

```
prof_forward_memory.py:96   prof_grpo_step.py:206      prof_backward_ops.py:489
probe_pad_histogram.py:61   recapture_correctness.py:52  recapture_arms.py:90
```

A peer's enumeration reported the tree already used `choices=` everywhere and that only the
new script lacked it. That is true of `cli.py`'s own parsers (`:1043`, `:1124`) and of the two
scripts that call `config.qwen36_27b()` directly (`profile_slice.py:214`,
`real_ckpt_smoke.py:20` — safe because they never reach this dispatch), and false of the six
above. Enumerating `_build_model(` callers and then checking each for `choices=` is what
separated them; grepping for `"--model"` alone conflates sites that reach the dispatch with
sites that do not.

So a per-script fix would have needed six edits and would have left the seventh script's
author to rediscover it.

## Fix

The refusal goes in `_build_model`, against `MODEL_NAMES` — one tuple that both the dispatch
and `cli.py`'s two `choices=` lists now read, so a new model cannot be added to one and missed
in the other. The message names the actual route for a local checkpoint, since passing its
path is the mistake most likely to be made:

```
unknown model 'qwen38_27b'; expected one of tiny, tiny-agent, qwen38-27b. A local 27B
checkpoint is selected with --model qwen38-27b plus TILERL_QWEN38_SOURCE=<dir>, not by
passing its path here
```

**Two mutants, both dead, and the second is the point.** Removing the guard fails the test.
Narrowing it to `("qwen38-27b",)` — so the guard refuses `tiny` and `tiny-agent` too — also
fails, caught by the negative control that every live name still builds *the config it names*
(`tiny-agent` is checked on `max_position_embeddings == 65536`, because it and `tiny` differ in
nothing else and a name that resolved to plain `tiny` would otherwise pass).

While writing that control I read `tiny(65536)`'s argument as the vocab size and printed
`vocab 320` as evidence `tiny-agent` was broken. It is `max_position_embeddings`
(`config.py:153`); the name resolves correctly. A one-line probe of the field I actually meant
settled it before the claim left the session.

497 passed, ruff clean.

## Rule

A dispatch's default branch must be an error, not one of its cases. When the default is a
working object, every wrong input produces a right-looking result, and the failure is not
observable at any call site — which is also why the fix belongs at the dispatch and not in the
callers.

And when enumerating call sites, enumerate the ones that reach the mechanism. `grep '"--model"'`
answers "who has this flag"; the question was "who reaches this dispatch", and the two sets
differ by six.
