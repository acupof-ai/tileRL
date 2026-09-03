---
question: Three `test_e2e.py` training tests fail on CUDA and pass on CPU. Did tonight's six merges break them?
source: H20 sm90, tilelang 0.1.13 (/work/tl013), torch 2.11.0; a702c9a and a4c2955 in separate pod checkouts; fixes verified on 9e3836b, suite sharded across five idle cards
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

## Fix (applied)

- `test_tape_gradcheck`: skipped on cuda, with the 1.9e-2 measurement in the
  skip message. There is no step size that works — the cast is unconditional,
  so any step small enough to be a derivative is below one bf16 ulp and any
  step large enough to survive the cast is not a derivative. The cuda tape is
  covered by `tests/test_ops_parity.py`.
- `reference.state_scatter`: `.to(device=..., dtype=...)` on both arms. The
  non-stepped path was also dropping `new_window` outright, which nothing
  covered.
- `test_opd_ema_self_teacher_shares_the_model`: **the third diagnosis above was
  right about the mechanism and wrong about where the fix goes.** Re-pointing
  `trainable` at the materialized tensors — the first suggestion — was tried and
  is wrong: `opd_loop` swaps EMA and student weights by copying *into* the live
  adapter objects because the engine holds those objects, so rebinding the dict
  detaches the engine from the tensors being trained, and the next step raises a
  device mismatch instead. The real defect was caller order. `build_engine` is
  what materializes; this test called `add_lora` before it, and every other
  caller in the tree already builds the engine first (`cli.py:191` says so in a
  comment). One line moved in the test; `train.py` unchanged.

## What the third test was hiding

With the ordering fixed it failed again, on an assertion that had never been
reachable: `moved <= set(trainable)`, "the quantized base does not move". On
sm90 it does. `Backend._served_fp4` twiddles `.wq` into the served layout **in
place** on first use and flags the tensor `_tl_twiddled` so `save_hf` can
untwiddle. Measured on H20 with a 4-token generation and no training at all:

```
after build_engine: same object False   bytes equal True
after generate:     same object True    bytes equal False
max abs delta 252.0, 4075 of 4096 bytes changed
```

That is by design and `save_hf` handles it, but "the frozen base stays
bit-identical" is false on sm90 and the assertion now allows exactly the
flagged keys. Worth knowing before anyone hashes weights to check a base is
frozen: on this card, serving one request changes them.

## A fourth, on CPU

`test_seedless_requests_decorrelate` failed on `ubuntu-latest` while passing on
`macos-14` in the same run. Six seedless completions came back byte-identical —
with all six seeds distinct, since they come from `secrets.randbits(31)`. The
tiny model at `T=0.7, top_p=0.8, top_k=20` is peaked enough that the draw is
deterministic in practice, so the test was asserting a property of the model's
entropy, not of the sampling stream. It now asserts the seeds. Reintroducing
the `seed=0` default fails it.

## Still red, and not ours

`test_fused_projections_parity.py::test_fused_fp8_qkvz_parity` and
`test_weights.py::test_fused_projections_parity` fail on clean `origin/main` on
the same card. The first never creates `layers.1.qkvz`; the second's fused
logits differ from unfused by 6.37 at `atol=0.01`. Both pass on CPU. Separate
investigation — recorded here so the next full cuda run is not read as a
regression.

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
