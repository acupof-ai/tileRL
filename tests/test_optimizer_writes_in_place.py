"""Every optimizer writes its parameter in place.

`_require_on_policy` (train.py:345) lets a training engine KEEP its captured decode
graphs across an optimizer step, and `invalidate_weights` (engine.py:1180) states the
premise: "every address a capture baked survives the update: `AdamW.step_one` and
`Adafactor.step_one` both end `p.copy_()` (in place)". A graph replays the addresses it
traced, so an optimizer that rebinds `p` instead of writing through it makes every
replayed rollout read pre-step weights -- silently, since nothing checks.

The premise was tested for AdamW only (test_decode_graph.py:386, via the cast refill).
It is asserted here for every implementation, ENUMERATED from `_Optimizer.__subclasses__`
rather than named: a hand-listed set silently omits the next optimizer, and this one
already had a third that the two named in the docstring did not cover (`iso.ISO`).

Run: TILERL_TARGET=cpu uv run pytest tests/test_optimizer_writes_in_place.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import torch

from tilerl.autograd import Adafactor, AdamW, _Optimizer
from tilerl.iso import ISO  # noqa: F401  -- imported so its subclass registers


def _implementations() -> list[type]:
    """Every `_Optimizer` subclass defining its own `step_one`, at any depth."""
    out, stack = [], list(_Optimizer.__subclasses__())
    while stack:
        c = stack.pop()
        stack.extend(c.__subclasses__())
        if "step_one" in c.__dict__:
            out.append(c)
    return sorted(out, key=lambda c: c.__name__)


def _build(cls: type) -> _Optimizer:
    """ISO wraps a base; the others take none. Kept here so a new optimizer needing
    construction args fails loudly at this line rather than being skipped.

    `begin()`, not `_step = 1`: ISO forwards it to its base, and Adafactor's
    `_step**decay_power` divides by zero at step 0. Poking the attribute leaves a
    wrapper's base at 0 and the arm dies before it asserts anything.
    """
    opt = cls(base=AdamW(lr=0.5)) if cls is ISO else cls(lr=0.5)
    opt.begin()
    return opt


def test_every_optimizer_writes_its_parameter_in_place():
    names = [c.__name__ for c in _implementations()]
    # Enumeration, not a name list: 0 or 1 would mean the walk found nothing and every
    # assertion below would be vacuous. Three exist today.
    assert len(names) >= 3, f"the subclass walk found {names}; it cannot have missed all"

    for cls in _implementations():
        # 2D and square: ISO reparameterizes 2D weights by their SVD and delegates
        # anything else to its base, so a 1D param would test AdamW three times.
        #
        # bfloat16, NOT float32, and this decides whether the test can fail at all:
        # every step_one computes `p32 = p.to(torch.float32)`, and on an f32 param that
        # RETURNS `p` ITSELF. The in-place update then lands on the parameter whatever
        # the final line does, so `p.data = ...` passes both assertions below. Measured:
        # the mutant was green until this dtype changed.
        p = torch.randn(8, 8, dtype=torch.bfloat16)
        g = torch.randn(8, 8, dtype=torch.bfloat16)
        before_ptr, before_val = p.data_ptr(), p.clone()

        _build(cls).step_one(p, g)

        assert p.data_ptr() == before_ptr, (
            f"{cls.__name__}.step_one rebound the parameter; every address a captured "
            f"graph baked now points at pre-step weights, and nothing raises"
        )
        assert not torch.equal(p, before_val), (
            f"{cls.__name__}.step_one left the parameter unchanged, so the address "
            f"assertion above proves nothing"
        )


def test_iso_delegates_non_2d_and_still_writes_in_place():
    """`ISO.step_one` returns early for a non-2D param (iso.py:101), handing it to the
    base. That branch never touches `p` itself, so the premise rests on the base -- which
    the loop above tests, but not on this path.
    """
    p = torch.randn(16, dtype=torch.bfloat16)
    g = torch.randn(16, dtype=torch.bfloat16)
    ptr, before = p.data_ptr(), p.clone()

    opt = ISO(base=Adafactor(lr=0.5))
    opt.begin()
    opt.step_one(p, g)

    assert p.dim() == 1, "this arm must run the delegating branch, not the SVD one"
    assert p.data_ptr() == ptr, "the delegated update rebound the parameter"
    assert not torch.equal(p, before), "the delegated update did nothing; the arm is vacuous"


if __name__ == "__main__":
    test_every_optimizer_writes_in_place = test_every_optimizer_writes_its_parameter_in_place
    test_every_optimizer_writes_in_place()
    test_iso_delegates_non_2d_and_still_writes_in_place()
    print("ok:", [c.__name__ for c in _implementations()])
