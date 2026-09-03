---
question: Three `test_e2e.py` training tests fail on CUDA and pass on CPU. Did tonight's six merges break them?
source: H20 sm90, tilelang 0.1.13 (/work/tl013), torch 2.11.0, GPU 1 idle; a702c9a and a4c2955 in separate pod checkouts
---

# Three CUDA-only training tests were red before tonight's merges, for three unrelated reasons

`test_tape_gradcheck`, `test_lora_train_step_on_frozen_fp4_base` and
`test_opd_ema_self_teacher_shares_the_model` fail under `TILERL_TARGET=cuda`
and pass under `TILERL_TARGET=cpu`. Found while gating PR #12; not caused by it.

**Not a regression.** All three fail identically on `a702c9a` — the `main` from
before #6, #7, #9, #10, #11 and #14 landed — in its own clean pod checkout:

```
tilelang 0.1.13 /work/tl013/bin/python3
checkout=a702c9a
3 failed, 35 deselected
```

Same three names, same three error shapes as on `a4c2955`. The hypothesis that
#11 (which changed the GDN prefill kernels and flipped the sm90 default) or #7
(which appends `hidden_out` inside the layer loop) broke the tape is wrong: the
failures predate both.

They are also not one bug. They are three.

## Root causes

**1. `test_tape_gradcheck` — the test cannot pass on CUDA by construction.**

```
AssertionError: w_norm: tape grad mismatch, max abs diff 8.17e-01
  analytic [-0.4783, 0.2226, 0.1226, 0.1030, 0.4745, 1.1581, 0.1349, 0.1941]
  numeric  [-0.6719, 0.2656, 0.1493, 0.5140, 0.4764, 0.3414, 0.1817, 0.0339]
```

The docstring says "in f32 (bf16 swamps a 1e-3 step)" and the test duly builds
f32 tensors — but on CUDA `Backend._rows` casts every activation to bf16
regardless of what the caller passed (`io = torch.bfloat16 if
self.target.startswith("cuda")`). Measured on the card:

```
_rows() casts a float32 activation to: torch.bfloat16
bf16 eps = 7.812e-03; a 1e-3 step on a ~1.0 weight is 0.1 bf16 ulps
linear() f32 in -> f32 out; max |y - ref| = 1.925e-02   (test atol = 5e-4)
```

The finite-difference perturbation is a tenth of one bf16 ulp, so `w + 1e-3`
rounds back to `w` and the *numeric* gradient is noise. The analytic one is
probably fine. This is a test-target problem, not a tape bug.

**2. `test_lora_train_step_on_frozen_fp4_base` — a missing device migration.**

```
RuntimeError: Expected all tensors to be on the same device, but found at least
two devices, cuda:0 and cpu!   at reference.py:850
```

`reference.state_scatter` migrates its `slots` index (`torch.as_tensor(...,
device=states.device)`) but casts the payload with dtype only:

```python
states[slots, layer_idx] = new_state.to(states.dtype)   # .to(dtype), never .to(device)
```

Same class as the `paged_attention` decode-arm bug PR #10 fixed: one arm of an
op migrates, the neighbouring line trusts its caller.

**3. `test_opd_ema_self_teacher_shares_the_model` — tape identity, and the
assertion already names it.**

```
AssertionError: train_step: tape produced no parameter gradients — ...
materialize() rebuilds any param whose device/dtype differs, and the new object
has a new id()                                          at train.py:100
```

`Backend.materialize` (`backend.py:453`) promises "never rewrites an existing
tensor: optimizer moments are keyed by `id(param)`" — and keeps that promise
for the tensors it does not have to move. On CUDA the params arrive on CPU, so
migration allocates new objects, and the `param_ids` the caller captured before
`materialize` no longer intersect what the tape recorded. On CPU nothing
migrates, every `id()` survives, and the test passes.

## Reproduction

```bash
REMOTE_DIR=/work/tilerl-check scripts/pod_sync.sh run check \
  'export PATH=/work/tl013/bin:$PATH
   python3 -c "import tilelang; print(tilelang.__version__)"
   CUDA_VISIBLE_DEVICES=<idle> python3 -m pytest tests/test_e2e.py -q --tb=line \
     -k "tape_gradcheck or lora_train_step_on_frozen_fp4_base or opd_ema_self_teacher_shares_the_model"'
```

`export PATH=/work/tl013/bin:$PATH` is load-bearing on any checkout older than
#11: the container's own python has tilelang 0.1.8, which cannot compile the
fp4/fp8 cells at all and fails eight *unrelated* parity tests with
`tl::tma_load` signature errors. Print the version before trusting any result
from this pod.

## Fix

Not applied — three unrelated causes in three subsystems, none of them the PR
this was found under.

- `test_tape_gradcheck`: mark it CPU-only, or give the CUDA arm a step and
  tolerance bf16 can actually resolve. A gradcheck at 0.1 ulps measures
  rounding.
- `reference.py:850` (and the `steps` branch above it): `.to(states.device,
  states.dtype)`, matching what the `slots` line already does.
- `materialize` / `train.py:100`: capture `param_ids` from the materialized
  dict, or have `materialize` migrate in place so `id()` survives. The
  assertion text is already the diagnosis; it just needs someone to act on it.

## Rule

CI is `TILERL_TARGET=cpu` on ubuntu and macos. Every CUDA-gated line in the
suite is unobserved unless a person runs it on a card, and three tests have
been red across an unknown number of commits without anyone noticing. This is
the second such entry today — see
[2026-09-03-cuda-suite-red-on-main.md](2026-09-03-cuda-suite-red-on-main.md),
whose two tests are now fixed. Two entries about the same blind spot is the
signal that the blind spot, not the individual tests, is the thing to fix.

"Six PRs merged and now three tests are red" is a hypothesis, not a finding.
Run the pre-merge commit in its own checkout before bisecting; here it cost one
pod round trip and saved a four-point bisect that would have found nothing.

A red set is not one bug. These three shared a symptom class (CUDA-only,
training path) and had nothing else in common — a shared cause was the
tempting story and it was wrong.
