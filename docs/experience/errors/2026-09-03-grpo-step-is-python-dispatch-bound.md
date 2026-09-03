---
question: The 27B GRPO recipe stopped OOMing. Why has step 1 not printed after 72 minutes?
source: H20 sm90 GPU 4, tilelang 0.1.13 (/work/tl013), torch 2.11.0+cu129, Qwen3.8-27B-NVFP4, grpo-gsm8k-27b group=8 max_new_tokens=256; py-spy 60 s at 50 Hz, 2999 samples, pid verified through /proc/<pid>/cmdline
---

# The 27B GRPO step is bound by per-call Python dispatch, not by memory or by compilation

PR #15 took the step peak from an OOM at 95.04 GiB to 33.53 GiB, and the recipe
now runs. It has not produced a step. 72 minutes of process time, memory flat at
41.3 GiB, and the training banner long past.

Two explanations were offered before anything was measured — that it was still
compiling, and that it was stuck in the rollout. Both are wrong, and one
`py-spy dump` is not enough to tell: two single dumps taken minutes apart landed
in `gdn_backward` and in `linear_fp8` respectively, which is how a sampling
artifact looks when n=1.

## Where the time goes

60 s at 50 Hz, 2999 samples, on the live process:

| self | |
|---:|---|
| **14.77%** | `torch.nn.functional.pad` |
| 12.64% | `reference._gdn_chunk_bwd` |
| 7.50% | `reference._gdn_chunk_fwd` |
| 6.07% | `torch.nn.Module.__call__` |
| 5.94% | `backend.linear` |
| 4.57% | `tvm_ffi.func` |
| 3.63% | `tilelang builder._parse_phase2_key` |

| inclusive | |
|---:|---|
| **60.99%** | rollout (an `Engine.step` frame on the stack) |
| **33.74%** | tape backward |
| 29.64% | tilelang jit argument binding |
| 28.41% | `tilerl_kernels/reference.py`, the torch-eager cells |
| 16.71% | tilelang eager dispatch |

Rollout and backward are 61/34, so neither guess was right on its own. The
compilation theory is also dead on its own terms: the TileLang compile count had
been flat at 702 for three minutes before the profile started.

## `_pad2d` is 14.8% and it is not a no-op

`_pad2d` returns its argument unchanged when nothing needs padding
(`backend.py:40-42`), so every one of those samples is a real allocate-and-copy.
By call site:

| | |
|---:|---|
| 2.70% | `_base_linear` -> `linear_fp4` |
| 2.63% | `_gdn` -> `linear` |
| 2.43% | `_base_linear` -> `linear_fp8` |
| 1.97% | `_add_via` -> `linear` |
| 1.93% | `_mlp_body` -> `linear` |
| 1.00% | `linear_bwd` |

It is per call, not per load: the pad is recomputed on every linear in every
layer of every forward.

## What this costs

At 100 steps the recipe cannot finish. Step 1 has not completed in 72 minutes;
even taking that as the whole step, 100 steps is 120 hours. The run is worth
leaving up only until step 1 prints, because that number has never existed —
nothing had previously survived the OOM to walk the 27B training backward.

## Rule

A stall gets a profile before it gets an explanation. Two people produced two
confident diagnoses here — "still compiling" and "stuck in the rollout" — from
`tail` and from single `py-spy dump`s, and the 60 s profile agreed with neither.
A single stack sample is one sample; it says where the process was, not where it
spends its time.

The second rule is narrower. `# ponytail: torch-eager backward, tilelang kernel
when perf demands` is now cashed: perf demands. 28.4% of the step is in the
torch-eager reference cells, and the GDN backward is the largest single block in
it.
