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

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tilerl.autograd import Adafactor
from tilerl.cli import _build_model
from tilerl.engine import SamplingParams, build_engine
from tilerl.eval import MATCHERS
from tilerl.iso import ISO
from tilerl.kv_cache import NoPrefixStore
from tilerl.testing import RefBackend
from tilerl.train import grpo_loop, train_step

#: Defaults are the CPU fixture this was written against, and that fixture is VOID for the
#: 2.7x question -- both voids are recorded in
#: errors/2026-09-08-the-first-steps-to-target-and-the-gate-that-voided-it.md. They are here
#: so the script runs with no arguments; the real run overrides all of them.
DEFAULTS = dict(model="tiny", sft_steps=40, rl_steps=24, group=6, max_new_tokens=6,
                sft_lr=3e-2, rl_lr=1e-2, seed=0)


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULTS["model"],
                   help="passed to _build_model: 'tiny', or a real checkpoint name")
    p.add_argument("--reward", default="token-rate", choices=("token-rate", "gsm8k"),
                   help="token-rate is the CPU fixture and saturates in 4 steps; gsm8k "
                        "needs --data and a tokenizer, which means a real model")
    p.add_argument("--data", default="", help="jsonl of {prompt, answer} for --reward gsm8k")
    p.add_argument("--matcher", default="number", choices=list(MATCHERS),
                   help="how a gsm8k completion is scored")
    p.add_argument("--sft-steps", type=int, default=DEFAULTS["sft_steps"],
                   help="0 skips the SFT phase and runs RL from the init as built")
    p.add_argument("--rl-steps", type=int, default=DEFAULTS["rl_steps"])
    p.add_argument("--group", type=int, default=DEFAULTS["group"])
    p.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    p.add_argument("--sft-lr", type=float, default=DEFAULTS["sft_lr"])
    p.add_argument("--rl-lr", type=float, default=DEFAULTS["rl_lr"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    a = p.parse_args(argv)
    # Before the SFT phase, not inside the first RL arm: a missing --data used to surface
    # after 40 SFT steps had already run.
    if a.reward == "gsm8k" and not a.data:
        p.error("--reward gsm8k needs --data <jsonl of {prompt, answer}>")
    return a


def sft_base(cfg, model, backend, a):
    """A base whose spectrum carries something, so freezing it is not freezing noise."""
    rng = np.random.default_rng(a.seed)
    opt = Adafactor(lr=a.sft_lr)
    losses = []
    for _ in range(a.sft_steps):
        # Fresh batch per step: a fixed batch is memorized in 3-5 steps, which is
        # memorization rather than a trajectory (tests/test_iso.py:80 had this shape).
        losses.append(float(train_step(model, rng.integers(3, cfg.vocab_size, size=(8, 32))
                                       .astype(np.int64), backend, opt)))
    return losses


def spectra(params):
    return {k: torch.linalg.svdvals(v.detach().double())
            for k, v in params.items() if v.dim() == 2}


def drift(before, after):
    """Max and mean relative movement of the singular values, the validity gate."""
    rel = [float(((after[k] - s0).abs() / s0.clamp_min(1e-12)).max()) for k, s0 in before.items()]
    return max(rel), sum(rel) / len(rel)


def make_reward(cfg, a):
    """``(reward_fn, prompts)``. Two shapes, and only one of them can carry the 2.7x."""
    if a.reward == "token-rate":
        half = cfg.vocab_size // 2

        def rate(prompt, completion):  # a rate: its expectation does not grow with length
            return sum(1 for t in completion if t < half) / max(len(completion), 1)

        return rate, [[1, 2, 3, 4]]

    # gsm8k: correctness, which has the headroom a token rate lacks. It needs a tokenizer,
    # so it needs a real model -- refused rather than faked, because a stub tokenizer would
    # produce a curve of the stub. `--data` is validated in `_args`, before the SFT phase.
    from tilerl.tokenizer import get_tokenizer

    tok = get_tokenizer(None if a.model == "tiny" else a.model)
    rows = [json.loads(ln) for ln in Path(a.data).read_text().splitlines() if ln.strip()]
    match = MATCHERS[a.matcher]
    gold, prompts = {}, []
    for r in rows:
        ids = tuple(tok.encode(r["prompt"]))
        gold[ids] = r["answer"]
        prompts.append(list(ids))

    def correct(prompt, completion):
        text = tok.decode([int(t) for t in completion])
        return float(match(text, gold[tuple(int(t) for t in prompt)]))

    return correct, prompts


def rl_arm(cfg, model, make_opt, a):
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                          decode_graph=False, prefix_store=NoPrefixStore())
    reward, prompts = make_reward(cfg, a)
    before = spectra(model.params)
    hist = list(grpo_loop(engine, model, prompts, reward, a.rl_steps, backend,
                          make_opt(), group=a.group,
                          sampling=SamplingParams(max_new_tokens=a.max_new_tokens),
                          seed=a.seed))
    rewards = [h[0] for h in hist]
    return rewards, drift(before, spectra(model.params))


def steps_to(rewards, target):
    for i, r in enumerate(rewards, 1):
        if r >= target:
            return i
    return None


def main(argv=None):
    a = _args(argv)
    cfg, base = _build_model(a.model, seed=a.seed, keep_master=True)
    backend = RefBackend()
    torch.manual_seed(a.seed)
    if a.sft_steps:
        sft = sft_base(cfg, base, backend, a)
        print(f"SFT base: loss {sft[0]:.3f} -> {sft[-1]:.3f} over {a.sft_steps} steps "
              f"(fresh batch each step)")
    else:
        print("SFT phase skipped (--sft-steps 0): Sigma is the init's, which carries no "
              "information on a random build")
    print(f"model {a.model}  reward {a.reward}  group {a.group}  "
          f"max_new_tokens {a.max_new_tokens}  rl_steps {a.rl_steps}\n")

    arms = {"Adafactor": lambda: Adafactor(lr=a.rl_lr),
            "ISO(Adafactor)": lambda: ISO(Adafactor(lr=a.rl_lr))}
    out = {}
    for name, mk in arms.items():
        model = type(base)(cfg, {k: v.clone() for k, v in base.params.items()})
        out[name] = rl_arm(cfg, model, mk, a)
        r, (dmax, dmean) = out[name]
        print(f"{name:>16}: reward {r[0]:.3f} -> {r[-1]:.3f}   "
              f"Sigma drift max {100 * dmax:6.2f}%  mean {100 * dmean:5.2f}%")

    print(f"\n{'target reward':>16} " + "  ".join(f"{n:>14}" for n in arms))
    base_r = out["Adafactor"][0]
    for target in (np.mean(base_r[:3]) + d for d in (0.05, 0.10, 0.15)):
        cells = [steps_to(out[n][0], target) for n in arms]
        print(f"{target:>16.3f} " + "  ".join(
            f"{(str(c) + ' steps') if c else 'not reached':>14}" for c in cells))

    # Void 2, checked before the ratio is read: a saturated reward makes the targets
    # cluster, since every one of them lands in the pre-plateau steps. It does not
    # necessarily make them indiscriminate -- say which, rather than overstating.
    tail = base_r[len(base_r) // 2:]
    if max(tail) - min(tail) < 1e-9:
        span = [steps_to(base_r, np.mean(base_r[:3]) + d) for d in (0.05, 0.10, 0.15)]
        resolved = len({s for s in span if s is not None})
        print(f"\nSATURATED: the reward is flat over the last {len(tail)} steps at "
              f"{tail[-1]:.3f}, so every target sits in the ramp before the plateau"
              + (f" and {resolved} distinct step counts separate them -- the ratio is a "
                 f"measurement of the ramp's slope, not of a trajectory."
                 if resolved > 1 else
                 " and they all resolve to one step count -- no target discriminates."))

    # Void 1, the validity gate: ISO's premise is a preserved spectrum. If Sigma moves in the
    # RL phase, this fixture does not reproduce the paper's condition and no step ratio here
    # carries the claim -- report that instead of the ratio.
    iso_max = out["ISO(Adafactor)"][1][0]
    ada_max, ada_mean = out["Adafactor"][1]
    print(f"\nvalidity gate: ISO Sigma drift max {100 * iso_max:.2f}%")
    print("  ISO freezes Sigma by construction, so its own drift is the FLOOR of the"
          " measurement's\n  precision, not evidence about the premise. The premise is"
          " tested by the FREE arm:")
    print(f"  Adafactor (free spectrum) drift max {100 * ada_max:.2f}% mean {100 * ada_mean:.2f}%")
    verdict = ("VOID: a free optimizer moves the spectrum, so RLVR does not preserve it here"
               if ada_max > 0.05 else
               "OK: the free arm holds the spectrum, so the paper's condition is reproduced")
    print(f"  {verdict} (threshold 5% max relative movement)")


if __name__ == "__main__":
    main()
    # A runnable check on the gate's own logic rather than on the measurement:
    # ISO must hold its spectrum to float precision, or `frames` is not freezing Sigma.
    _cfg, m = _build_model("tiny", seed=0, keep_master=True)
    s0 = spectra(m.params)
    opt = ISO(Adafactor(lr=1e-1))
    rng = np.random.default_rng(0)
    for _ in range(3):
        train_step(m, rng.integers(3, _cfg.vocab_size, size=(4, 16)).astype(np.int64),
                   RefBackend(), opt)
    dmax, _ = drift(s0, spectra(m.params))
    assert dmax < 5e-2, f"ISO moved its frozen spectrum by {100 * dmax:.2f}%"
    print(f"\nself-check: ISO holds Sigma to {100 * dmax:.3f}% over 3 steps at lr=0.1")
