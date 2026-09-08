"""Steps to a target reward: ISO vs Adafactor, with the Sigma-drift validity gate.

The project measures `seconds_per_step` (85.617 s, #273) and has never measured
`steps_to_score`, so the product it is optimizing has an unknown factor. This is the
instrument for that factor, at a scale where the loop closes in minutes.

`design-rl-stack.md:27` reports ~2.7x fewer steps for ISO on Qwen3-4B/8B under RLVR. The
claim's premise is that RLVR PRESERVES the base model's singular spectrum and moves only the
frames -- so ISO freezes Sigma from the base. Two consequences for any small-scale attempt:

  1. A random init has no informative spectrum. Freezing it locks a random constraint, so
     ISO there is a pure cost -- measured at 0.75-0.81x (ISO needing MORE steps) over four
     learning rates. That number does not refute 2.7x; it refutes using a random init to
     predict it.
  2. So the base must be SFT'd first, to put information in Sigma. Whether tiny's SFT
     spectrum resembles a pretrained one is not assumable -- it is MEASURABLE inside the
     experiment: record Sigma's drift during the RL phase. If Sigma moves, the paper's
     premise does not hold in this setting and the step ratio carries nothing. If Sigma is
     near-static, the condition is reproduced and the ratio means something.

That gate is the difference between a number and a number that measures what it claims.

Run: TILERL_TARGET=cpu uv run python scripts/steps_to_reward.py
"""

from __future__ import annotations

import numpy as np
import torch

from tilerl.autograd import Adafactor
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.iso import ISO
from tilerl.kv_cache import NoPrefixStore
from tilerl.testing import RefBackend
from tilerl.train import grpo_loop, train_step

SFT_STEPS = 40  # enough for the loss to leave its random-init plateau
RL_STEPS = 24
GROUP = 6


def sft_base(cfg, backend, seed=0):
    """A base whose spectrum carries something, so freezing it is not freezing noise."""
    _, model = _build_model("tiny", seed=0, keep_master=True)
    rng = np.random.default_rng(seed)
    opt = Adafactor(lr=3e-2)
    losses = []
    for _ in range(SFT_STEPS):
        # Fresh batch per step: a fixed batch is memorized in 3-5 steps, which is
        # memorization rather than a trajectory (tests/test_iso.py:80 had this shape).
        losses.append(float(train_step(model, rng.integers(3, cfg.vocab_size, size=(8, 32))
                                       .astype(np.int64), backend, opt)))
    return model, losses


def spectra(params):
    return {k: torch.linalg.svdvals(v.detach().double())
            for k, v in params.items() if v.dim() == 2}


def drift(before, after):
    """Max and mean relative movement of the singular values, the validity gate."""
    rel = [float(((after[k] - s0).abs() / s0.clamp_min(1e-12)).max()) for k, s0 in before.items()]
    return max(rel), sum(rel) / len(rel)


def rl_arm(cfg, model, make_opt, seed=0):
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                          decode_graph=False, prefix_store=NoPrefixStore())
    half = cfg.vocab_size // 2

    def reward(prompt, completion):  # a rate, so its expectation does not grow with length
        return sum(1 for t in completion if t < half) / max(len(completion), 1)

    before = spectra(model.params)
    hist = list(grpo_loop(engine, model, [[1, 2, 3, 4]], reward, RL_STEPS, backend,
                          make_opt(), group=GROUP,
                          sampling=SamplingParams(max_new_tokens=6), seed=seed))
    rewards = [h[0] for h in hist]
    return rewards, drift(before, spectra(model.params))


def steps_to(rewards, target):
    for i, r in enumerate(rewards, 1):
        if r >= target:
            return i
    return None


def main():
    cfg, backend = _build_model("tiny", seed=0, keep_master=True)[0], RefBackend()
    torch.manual_seed(0)
    base, sft = sft_base(cfg, backend)
    print(f"SFT base: loss {sft[0]:.3f} -> {sft[-1]:.3f} over {SFT_STEPS} steps "
          f"(fresh batch each step)\n")

    arms = {"Adafactor": lambda: Adafactor(lr=1e-2),
            "ISO(Adafactor)": lambda: ISO(Adafactor(lr=1e-2))}
    out = {}
    for name, mk in arms.items():
        model = type(base)(cfg, {k: v.clone() for k, v in base.params.items()})
        out[name] = rl_arm(cfg, model, mk)
        r, (dmax, dmean) = out[name]
        print(f"{name:>16}: reward {r[0]:.3f} -> {r[-1]:.3f}   "
              f"Sigma drift max {100 * dmax:6.2f}%  mean {100 * dmean:5.2f}%")

    print(f"\n{'target reward':>16} " + "  ".join(f"{n:>14}" for n in arms))
    base_r = out["Adafactor"][0]
    for target in (np.mean(base_r[:3]) + d for d in (0.05, 0.10, 0.15)):
        cells = [steps_to(out[n][0], target) for n in arms]
        print(f"{target:>16.3f} " + "  ".join(
            f"{(str(c) + ' steps') if c else 'not reached':>14}" for c in cells))

    # The validity gate: ISO's premise is a preserved spectrum. If Sigma moves in the RL
    # phase, this fixture does not reproduce the paper's condition and no step ratio here
    # carries the claim -- report that instead of the ratio.
    iso_max = out["ISO(Adafactor)"][1][0]
    print(f"\nvalidity gate: ISO Sigma drift max {100 * iso_max:.2f}%")
    print("  ISO freezes Sigma by construction, so its own drift is the FLOOR of the"
          " measurement's\n  precision, not evidence about the premise. The premise is"
          " tested by Adafactor's drift:")
    ada_max, ada_mean = out["Adafactor"][1]
    print(f"  Adafactor (free spectrum) drift max {100 * ada_max:.2f}% mean {100 * ada_mean:.2f}%")
    print("  If a FREE optimizer barely moves Sigma, RLVR preserves the spectrum here and the"
          "\n  paper's condition is reproduced. If it moves Sigma a lot, it is not.")


if __name__ == "__main__":
    main()
    # A runnable check on the gate's own logic rather than on the measurement:
    # ISO must hold its spectrum to float precision, or `frames` is not freezing Sigma.
    _cfg = _build_model("tiny", seed=0, keep_master=True)[0]
    _, m = _build_model("tiny", seed=0, keep_master=True)
    s0 = spectra(m.params)
    opt = ISO(Adafactor(lr=1e-1))
    rng = np.random.default_rng(0)
    for _ in range(3):
        train_step(m, rng.integers(3, _cfg.vocab_size, size=(4, 16)).astype(np.int64),
                   RefBackend(), opt)
    dmax, _ = drift(s0, spectra(m.params))
    assert dmax < 5e-2, f"ISO moved its frozen spectrum by {100 * dmax:.2f}%"
    print(f"\nself-check: ISO holds Sigma to {100 * dmax:.3f}% over 3 steps at lr=0.1")
