# A dtype chosen for convenience made the mechanism under test dead code — 2026-09-08

**Status:** fixed in the same commit (`tests/test_optimizer_writes_in_place.py`).

## Context

`_require_on_policy` (`train.py:345`) lets a training engine keep its captured decode
graphs across an optimizer step. The premise, stated in `invalidate_weights`
(`engine.py:1180`): every address a capture baked survives the update, because each
`step_one` ends in `p.copy_()`. An optimizer that rebinds `p` instead would make every
replayed rollout read pre-step weights, silently.

Only AdamW was covered. The task was to enumerate all implementations and assert the
in-place write, with a mutation check that `p.copy_(x)` → `p.data = x` reddens it.

The test was written, it passed, and the mutation **also** passed — on all three
optimizers.

## Root cause

The parameter in the fixture was `torch.float32`. Every `step_one` opens with

```python
p32 = p.to(torch.float32)
```

and on an f32 tensor `.to(torch.float32)` returns **the same object**:

```
f32  p.to(torch.float32) is p   -> True    same data_ptr
bf16 b.to(torch.float32) is b   -> False
```

So `p32` *was* `p`. Every `addcdiv_` / `mul_` / `sub_` in the body wrote straight through
the parameter, and the final line — the only line the test exists to check — was dead
code. `data_ptr()` unchanged: true either way. Value changed: true either way. Both
assertions green, mechanism unexercised.

The dtype was chosen to avoid bfloat16 rounding noise in the value assertion. It has no
apparent relationship to *whether a write is in place*.

## Fix

The fixture is bfloat16, and the docstring says why in the same breath as the choice, so
the next person does not "simplify" it back. All three mutants then redden:

| mutant | result |
|---|---|
| `AdamW.step_one` rebinds | 1 failed |
| `Adafactor.step_one` rebinds | 2 failed |
| `ISO.step_one` rebinds | 1 failed |

A second preparation step had the same shape and was caught the same way. The fixture
first advanced the step counter with `opt._step = 1`; `ISO` forwards `begin()` to its base
(`iso.py:62`), so poking the attribute left the base at `_step = 0` and Adafactor's
`1.0 - self._step**self.decay_power` (`:574`) raised `ZeroDivisionError`. **Setting
internal state to build a fixture bypassed the very forwarding the fixture existed to
cover.** `opt.begin()` instead.

## Rule

**A setting in a test that looks unrelated to the property under test can short-circuit
the mechanism entirely.** Not a parameter, not a mode — a dtype. Ask of every fixture
value: is there a path by which this makes the line I am testing unreachable? Here the
chain was three steps long (f32 → `.to()` is identity → `p32` aliases `p` → in-place work
lands on the parameter → the final write is redundant), and no step of it is visible from
the assertions.

**When a mutation comes back green, find out why before recording the result.** The
mutation ran, it passed, and "the mutant passed" is one keystroke from "the test is
covered." Chasing the green is what found the aliasing. Had it been recorded instead, the
delivered test would have covered three optimizers, looked stricter than what it replaced,
and checked nothing — and its apparent strictness would have discouraged the next reader
from examining it.

**Preparation code is under test too.** Both defects here were in the fixture, not the
assertions: one chose a dtype, one set an attribute, and each removed a path the test was
written to exercise. The nearest relative in the tree is
[the W=8 capture aborted on the test's own spy](2026-09-03-w8-capture-aborted-on-the-tests-own-spy.md),
where the instrument, not the system, produced the failure being read.
