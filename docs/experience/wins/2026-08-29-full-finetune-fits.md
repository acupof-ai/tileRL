# Full fine-tuning of the 27B fits one card: 73.2 of 95 GiB

## Context

Full-parameter training OOMed at 95.09 of 95.22 GiB inside the backward's
`gemm_tn`. The instinct was to hunt for the biggest tensor. The instinct was
wrong: the blockers are arithmetic, and adding up the ledger settles all three
before any measurement.

27B = **26.90 B parameters**, of which 25.62 B are quantized linears:

| | GiB |
|---|---:|
| bf16 master, all params | 50.1 |
| fp4/fp8 bytes + scales | 14.9 |
| `keep_master` = both resident | 65.0 |
| **Adam m+v, fp32** | **200.4** |
| all weight gradients coexisting, bf16 | 50.1 |
| Adafactor factored second moment | **0.03** |

No search through the tape can save 200 GiB. The card is 95.

## What Worked

**1. Adafactor instead of Adam (−200.4 GiB).** A 2D parameter's second moment
is stored as a row vector plus a column vector, and there is no first moment
(`beta1=0`). 0.03 GiB for the whole model. `Adafactor` has AdamW's `step`
signature so the loop does not care which it holds.

**2. Gradients consumed inside backward (−50.1 GiB).** What forced every weight
gradient to coexist was `clip_grad_norm`: a global norm has to see all of them
before any update can be scaled. Adafactor clips each update instead (RMS ≤ 1
— the paper's own reason for not needing global clipping), so a parameter can
be updated and its gradient freed the moment backward finalizes it. `_step`
takes that path for any optimizer declaring `streams`; LoRA keeps AdamW and the
global norm. The streaming and collecting paths are bit-identical where they
overlap (`test_adafactor_streaming_matches_collecting`) — updates are
independent per parameter, so replay order does not matter.

**3. Drop the served bytes once a master exists (−14.9 GiB).** With masters the
tape routes every linear through `RecordingBackend.master_linear`, a dense bf16
GEMM; the quantized bytes are never read. `drop_quantized()` frees them at the
training entry points — deliberately NOT in `load_hf`, so the loader stays a
bit-exact round trip.

## Measured (27B, 64 layers, H20 GPU 7, B=1 T=64)

| phase | live GiB | peak GiB |
|---|---:|---:|
| materialize (bf16 masters only) | 50.10 | 50.10 |
| forward (tape 1187 entries) | 51.80 | 51.80 |
| backward + streamed update | 50.37 | **73.20** |

Backward returns to 50.37 — 0.2 GiB above where it started — so **no gradient
outlives its own update**, which is the property the whole design turns on.
The tape is 1187 entries against LoRA's 2678: a master linear is one dense GEMM
where a LoRA linear records two more matmuls and an add.

## Rule

When a request is "make X fit", add up the ledger before profiling. Three of
these four terms are fixed by the parameter count and the optimizer's choice of
state — no amount of measurement moves them, and measurement would have found
the 4.7 GiB tensors first and missed the 200 GiB one entirely.
