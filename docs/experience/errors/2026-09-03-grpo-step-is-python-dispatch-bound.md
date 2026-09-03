---
question: The 27B GRPO recipe prints nothing for 72 minutes. Is it stuck, and where does the time go?
source: H20 sm90 GPU 4, tilelang 0.1.13 (/work/tl013), torch 2.11.0+cu129, Qwen3.8-27B-NVFP4, grpo-gsm8k-27b group=8 max_new_tokens=256; py-spy 60 s at 50 Hz, 2999 samples, pid verified through /proc/<pid>/cmdline
---

# The 27B GRPO step is ~72 s and dispatch-bound; the silence was the logging

PR #15 took the step peak from an OOM at 95.04 GiB to 33.53 GiB, and the recipe
now runs. For 72 minutes it printed nothing, memory flat at 41.3 GiB, the
training banner long past. It was at step 89 of 100.

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

## What this costs — corrected, and the correction is the point

**The run was never stalled.** `py-spy dump --locals` on the same pid read
`step: 89` out of 100, with `out` already holding 89 tuples of
`(reward, ce, secs, tied)` whose `secs` are 70.0 and 73.9. **~72 s per step,
100 steps in about two hours.**

Nothing printed because nothing prints until the end: `grpo_loop` accumulates
the whole history and returns it, and `cli.py:222` loops over the result to log
it. A two-hour run emits no progress line until it is over.

So the earlier reading of the silence — mine, in the first version of this
entry, and a second agent's before that — was wrong, and wrong in the same way
twice: an absent log line was read as an absent step. The profile above is
unaffected; it measured where the time goes, not how much there was.

The defect worth fixing is the logging, not the speed. A recipe that runs for
two hours and reports nothing until it exits cannot be watched, cannot be
stopped early on a bad curve, and produces exactly this class of mistake.

## Rule

A stall gets a profile before it gets an explanation, and **a stall gets
confirmed before it is called a stall**. Three diagnoses were offered here —
"still compiling", "stuck in the rollout", "step 1 has not finished in 72
minutes" — and all three were wrong. The first two came from `tail` and from
single `py-spy dump`s; the third came from me, from the absence of a log line.

`py-spy dump --locals` reads the loop counter straight out of the running
frame. It cost one command and it is the measurement all three guesses were
substituting for. Reach for it before describing a process as hung: silence in
a log is a fact about the logging.

The second rule is narrower. `# ponytail: torch-eager backward, tilelang kernel
when perf demands` is now cashed: perf demands. 28.4% of the step is in the
torch-eager reference cells, and the GDN backward is the largest single block in
it.
