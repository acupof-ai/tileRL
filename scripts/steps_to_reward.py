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

On a card, against the 27B (no SFT phase -- a pretrained spectrum already carries
information, which is what that phase exists to give tiny). `--data` is P1's own train file
so this curve is comparable with run 1's:

    scripts/pod_run.sh isorl <card> -- env TILERL_TARGET=cuda \\
      TILERL_QWEN38_SOURCE=<checkpoint dir> HF_ENDPOINT=https://hf-mirror.com \\
      python scripts/steps_to_reward.py --model qwen38-27b --reward gsm8k \\
      --data /work/p1_gsm8k_train.jsonl --sft-steps 0 --rl-steps N --group 8 \\
      --max-new-tokens 256

`HF_ENDPOINT` because huggingface.co does not resolve from the pod but the mirror does --
and NOT the offline switches, which would turn a cache miss into a failure instead of a
fetch. One endpoint being unreachable is not the box being offline.

Read `tied groups` on a short run before choosing `--rl-steps` or the cap: run 2 collapsed
onto its rollout cap at step 41 with the guard that ran before step 1 satisfied
(errors/2026-09-06-the-rollouts-grew-into-the-cap.md), so a cap that clears at step 1 is not
a cap that holds. High ties mean the cap or the reward, not ISO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

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

#: `_build_model` dispatches on the NAME and falls through to `tiny` for anything it does
#: not recognise (`cli.py:56,71`) -- so `--model /path/to/checkpoint` silently builds a
#: random 64-hidden 2-layer tiny and the whole run reads real. A local checkpoint arrives
#: through `TILERL_QWEN38_SOURCE`, which is what `--model qwen38-27b` already reads.
MODELS = ("tiny", "tiny-agent", "qwen38-27b")


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULTS["model"], choices=MODELS,
                   help="a name _build_model dispatches on; point qwen38-27b at a local "
                        "checkpoint with TILERL_QWEN38_SOURCE=<dir>")
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
    p.add_argument("--backend", default="auto", choices=("auto", "reference"),
                   help="auto takes the real backend (TILERL_TARGET); reference is the "
                        "torch-eager CPU twin, which is CPU no matter what TILERL_TARGET says")
    p.add_argument("--sigma-per-class", type=int, default=2,
                   help="matrices sampled per 2D shape class for the drift gate; the full "
                        "census is 1698 s per call on the 27B and 9.47 GiB at f64. Default 2 "
                        "rather than 1 so the within-class spread can be read at all -- with "
                        "one member per class the sample cannot audit itself")
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


def sigma_keys(params, per_class: int = 1):
    """A FIXED sample of 2D parameter names, ``per_class`` per shape class, sorted.

    The census is unaffordable: measured on the 27B at f32, one svdvals per shape class times
    the class size is 1698 s per call, so two calls per arm times two arms is 1.9 h of pure
    instrument (`probe_svd_cost.py`). At f64 it does not even fit -- 9.47 GiB for the
    248320x5120 embedding against 3.56 GiB free.

    Shape is the sampling unit because both the cost and the conditioning track it. Fixed and
    sorted matters more than which matrices: two arms sampled differently produce drifts that
    cannot be compared, and the gate would still print a number.
    """
    by_shape = {}
    for k, v in sorted(params.items()):
        if v.dim() == 2:
            by_shape.setdefault(tuple(v.shape), []).append(k)
    return [k for ks in by_shape.values() for k in ks[:per_class]]


def spectra(params, keys=None):
    """Singular values, f32, computed on the HOST.

    Three measurements pin every part of this. The verdict is a threshold on a percentage, so
    f64 buys nothing and cost 9.47 GiB on the first card run. Sampling is forced: a full sweep
    is ~1700 s per call. And the host is forced -- `materialize` has moved the params to the
    card by the time the gate reads them, leaving 3.56 GiB free, while ONE f32 embedding is
    4.74 GiB, so a per-shape-class sample necessarily includes it and OOMs anyway. The card
    buys 1.01-1.08x over the host on the large classes (SVD here is algorithm-bound, not
    bandwidth-bound), so `.cpu()` costs ~6% of an already-sampled sweep and zero card memory.
    ``.cpu()`` comes BEFORE ``.float()`` and the order is the whole fix: ``.float().cpu()``
    casts on the SOURCE device, so it allocated the f32 embedding on the card and OOMed asking
    for exactly 4.74 GiB — the same failure the f64 version had, one cast later. Copying the
    bf16 bytes first moves half as much and casts where there is room.
    """
    sel = params if keys is None else {k: params[k] for k in keys}
    return {k: torch.linalg.svdvals(v.detach().cpu().float())
            for k, v in sel.items() if v.dim() == 2}


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
    # `get_tokenizer` takes a hub id or a DIRECTORY, not a model name: "qwen38-27b" is not a
    # repo and 401s, so the name resolves only through `cli._qwen38_tokenizer`.
    from tilerl.cli import _qwen38_tokenizer
    from tilerl.tokenizer import get_tokenizer

    tok = _qwen38_tokenizer() if a.model == "qwen38-27b" else get_tokenizer(None)
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


def rl_arm(cfg, model, make_opt, a, backend, keys):
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                          decode_graph=False, prefix_store=NoPrefixStore())
    reward, prompts = make_reward(cfg, a)
    before = spectra(model.params, keys)
    hist = list(grpo_loop(engine, model, prompts, reward, a.rl_steps, backend,
                          make_opt(), group=a.group,
                          sampling=SamplingParams(max_new_tokens=a.max_new_tokens),
                          seed=a.seed))
    rewards = [h[0] for h in hist]
    # `tied` (h[3]) is the fraction of groups whose advantages are ALL zero, i.e. steps that
    # produced no gradient. It gates the Sigma reading: a run where every group ties leaves
    # Sigma at 0.00% because nothing was applied, and the drift gate then reads OK -- "the
    # spectrum is preserved" said about a model that was never trained.
    tied = sum(h[3] for h in hist) / len(hist)
    after = spectra(model.params, keys)
    d = drift(before, after)
    # Whether a per-class sample carries the census verdict cannot be settled on tiny: its
    # census argmax is `layers.0.o_proj`, the sole member of its shape class, so no sample can
    # exclude it and both a naive and an adversarial check came back unable to fail. So the
    # sample audits itself HERE, on the first arm, where the classes have 3-144 members: the
    # per-class spread of the drift says whether one member represents its class.
    spread = {}
    for k in before:
        spread.setdefault(tuple(model.params[k].shape), []).append(
            float(((after[k] - before[k]).abs() / before[k].clamp_min(1e-12)).max()))
    return rewards, d, tied, spread


def steps_to(rewards, target):
    for i, r in enumerate(rewards, 1):
        if r >= target:
            return i
    return None


def sigma_verdict(tied: float, ada_max: float) -> str:
    """The drift gate's three outcomes. `tied` comes first because it invalidates the other
    two: with every advantage zero no step was applied, so a 0% drift is arithmetic rather
    than a reading, and calling that OK is the gate answering a question it never tested."""
    if tied > 0.99:
        return "NOT TESTED: no gradient was applied, so a 0% drift says nothing"
    if ada_max > 0.05:
        return "VOID: a free optimizer moves the spectrum, so RLVR does not preserve it here"
    return "OK: the free arm holds the spectrum, so the paper's condition is reproduced"


def main(argv=None):
    a = _args(argv)
    # RefBackend is the torch-eager CPU reference (`device = cpu`, hardwired), so a card run
    # must take the real backend or every arm silently runs on the host.
    backend = get_backend() if a.backend == "auto" else RefBackend()
    cfg, base = _build_model(a.model, seed=a.seed, keep_master=True)
    # Full fine-tuning never reads the served bytes -- the tape routes every linear through
    # `master_linear` once a bf16 master exists -- and they are 16.88 GiB on the 27B (9.90 fp8
    # + 6.98 uint8, measured). `cli.py:275` and `:1094` do this at the training entry points;
    # this script builds its engine directly and so has to do it itself. Not calling it is
    # what left the arm 4.74 GiB short in step_one.
    if a.model == "qwen38-27b":
        from tilerl.model import drop_quantized

        drop_quantized(base)
    torch.manual_seed(a.seed)
    if a.sft_steps:
        sft = sft_base(cfg, base, backend, a)
        print(f"SFT base: loss {sft[0]:.3f} -> {sft[-1]:.3f} over {a.sft_steps} steps "
              f"(fresh batch each step)")
    else:
        print("SFT phase skipped (--sft-steps 0): Sigma is the init's, which carries no "
              "information on a random build")
    print(f"model {a.model}  reward {a.reward}  group {a.group}  backend {backend.name}  "
          f"max_new_tokens {a.max_new_tokens}  rl_steps {a.rl_steps}\n")

    arms = {"Adafactor": lambda: Adafactor(lr=a.rl_lr),
            "ISO(Adafactor)": lambda: ISO(Adafactor(lr=a.rl_lr))}
    # One arm at a time, and the base's tensors are SNAPSHOTTED rather than kept as a second
    # live model: `keep_master=True` on the 27B is 65.0 GiB resident
    # (wins/2026-08-29-full-finetune-fits.md), so base + one clone is 115 GiB against a 95.6
    # GiB card and the run OOMs before step 1. The snapshot is on the host, where 22 GB of
    # bf16 costs RAM rather than the card, and each arm restores into the SAME tensors.
    snapshot = {k: v.detach().to("cpu", copy=True) for k, v in base.params.items()}
    keys = sigma_keys(base.params, a.sigma_per_class)
    print(f"Sigma sample: {len(keys)} of "
          f"{sum(1 for v in base.params.values() if v.dim() == 2)} 2D params, "
          f"{a.sigma_per_class} per shape class, fixed and identical across arms")
    out = {}
    for name, mk in arms.items():
        with torch.no_grad():
            for k, v in base.params.items():
                v.copy_(snapshot[k])
        out[name] = rl_arm(cfg, base, mk, a, backend, keys)
        r, (dmax, dmean), tied, _ = out[name]
        print(f"{name:>16}: reward {r[0]:.3f} -> {r[-1]:.3f}   "
              f"Sigma drift max {100 * dmax:6.2f}%  mean {100 * dmean:5.2f}%  "
              f"tied groups {100 * tied:5.1f}%")

    print(f"\n{'target reward':>16} " + "  ".join(f"{n:>14}" for n in arms))
    base_r = out["Adafactor"][0]
    for target in (np.mean(base_r[:3]) + d for d in (0.05, 0.10, 0.15)):
        cells = [steps_to(out[n][0], target) for n in arms]
        print(f"{target:>16.3f} " + "  ".join(
            f"{(str(c) + ' steps') if c else 'not reached':>14}" for c in cells))

    # Void 0, and it comes FIRST because it invalidates the other two: if every group tied,
    # no gradient was applied, so Sigma cannot have moved and the drift gate reads OK about
    # a model that was never trained. Measured on tiny + gsm8k: tied 100%, drift 0.0000%,
    # gate "OK". A gate whose green is produced by the absence of the thing it measures.
    ada_tied = out["Adafactor"][2]
    if ada_tied > 0.99:
        print(f"\nVOID: {100 * ada_tied:.0f}% of groups tied -- every advantage in the group "
              f"was zero, so no step changed a weight. Reward, Sigma drift and every number "
              f"above describe the INIT, not a trajectory. A binary reward on a model that "
              f"never scores gives one group value and GRPO's within-group normalization "
              f"then yields zero: the reward needs a scale the policy can already move on.")

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
    print(f"  {sigma_verdict(ada_tied, ada_max)} (threshold 5% max relative movement)")

    # The sample auditing itself, on the free arm, where the shape classes have 3-144 members.
    # tiny cannot answer this: its census argmax is the sole member of its class, so a sample
    # there is a no-op and both a naive and an adversarial check came back unable to fail. A
    # class whose members disagree by more than the threshold means one member does not
    # represent it, and --sigma-per-class must go up before the verdict is trusted.
    spread = out["Adafactor"][3]
    worst = max(((max(v) - min(v), s, len(v)) for s, v in spread.items() if len(v) > 1),
                default=(0.0, None, 0))
    if worst[1] is None:
        print("  sample audit: every shape class has one member, so the sample IS the census")
    else:
        gap, shape, n = worst
        print(f"  sample audit: widest within-class drift spread {100 * gap:.2f}% on {shape} "
              f"({n} members sampled)")
        if gap > 0.05:
            print("    ABOVE the 5% threshold, so one member does not represent its class -- "
                  "raise --sigma-per-class and re-read the verdict")


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

    # The three real-model defects, each asserted where it was measured rather than read.
    # 1. `--model <path>` used to fall through to a random tiny (cfg.vocab_size 320 against
    #    the checkpoint's 248320) and every number read real.
    assert "/" not in "".join(MODELS), "MODELS must be names _build_model dispatches on"
    try:
        _args(["--model", "/data00/models/Qwen3.8-27B-NVFP4"])
        raise AssertionError("a checkpoint PATH was accepted and would build a random tiny")
    except SystemExit:
        pass
    # 2. `get_tokenizer("qwen38-27b")` is a hub lookup that 401s; the name resolves only
    #    through `cli._qwen38_tokenizer`. Checked by WHICH function the qwen branch calls --
    #    an earlier version of this assert grepped `make_reward`'s source for the name and
    #    the comment above satisfied it, so the reverted call passed.
    from tilerl import cli as _cli
    from tilerl import tokenizer as _tokmod
    from tilerl.tokenizer import ByteTokenizer

    called = []
    _real_q, _real_g = _cli._qwen38_tokenizer, _tokmod.get_tokenizer
    _cli._qwen38_tokenizer = lambda: (called.append("by-name"), ByteTokenizer())[1]
    _tokmod.get_tokenizer = lambda src=None: (called.append(f"hub:{src}"), ByteTokenizer())[1]
    try:
        _probe = Path("/tmp/_str_selfcheck.jsonl")
        _probe.write_text('{"prompt": "2+3=?", "answer": "5"}\n')
        make_reward(None, _args(["--model", "qwen38-27b", "--reward", "gsm8k",
                                 "--data", str(_probe)]))
    finally:
        _cli._qwen38_tokenizer, _tokmod.get_tokenizer = _real_q, _real_g
    assert called == ["by-name"], \
        f"the 27B tokenizer must resolve through cli._qwen38_tokenizer, got {called}"
    # 3. Every group tying leaves Sigma at 0.00%, which the drift gate used to call OK.
    assert sigma_verdict(1.0, 0.0).startswith("NOT TESTED"), "a no-gradient run read as OK"
    assert sigma_verdict(0.4, 0.0).startswith("OK"), "a real run must still reach a verdict"
    assert sigma_verdict(0.4, 0.2).startswith("VOID"), "a moving spectrum must still void"
    # 4. The gate must not allocate on the CARD. `materialize` leaves ~3.5 GiB free and one f32
    #    embedding is 4.74, so a card-side cast OOMs even sampled. Asserting the RESULT's
    #    device was not enough and cost a second card run: `.float().cpu()` returns a cpu
    #    tensor and still allocated 4.74 GiB on the card, because `.float()` runs on the
    #    source device. The quantity to assert is the card high-water mark across the call.
    _s = spectra(m.params, sigma_keys(m.params, 2))
    assert _s and all(v.device.type == "cpu" for v in _s.values()), \
        f"spectra returned non-CPU tensors: {[str(v.device) for v in _s.values()][:3]}"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        spectra(m.params, sigma_keys(m.params, 2))
        grew = torch.cuda.max_memory_allocated() - base_alloc
        assert grew < 1 << 20, (
            f"the drift gate allocated {grew / 2**20:.1f} MiB on the card; it must cast on "
            f"the host (.cpu() BEFORE .float()), or it OOMs beside a materialized 27B")
        _where = "verified: 0 card bytes"
    else:
        # Stated rather than silently skipped: on a CPU host there is no card allocation to
        # measure, so neither assert above can fail and this line is not evidence.
        _where = "vacuous on this CPU host"
    print("self-check: a checkpoint path is refused, the 27B tokenizer resolves by name, "
          f"a 100%-tied run reads NOT TESTED, and the drift gate is host-side ({_where})")
