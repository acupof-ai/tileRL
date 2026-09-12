"""GRPO: the policy-gradient step and the group baseline.

The step's gradient is the causal-CE gradient scaled per row by the advantage
and masked to completion tokens, so the sharp gate is algebraic — advantage 1
everywhere must reproduce an SFT step exactly, and advantage 0 must be a no-op.
The loop then has to actually raise reward on a task the tiny model can learn.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from tilerl.autograd import AdamW
from tilerl.cli import _build_model
from tilerl.eval import last_number
from tilerl.testing import RefBackend
from tilerl.train import group_advantages, rl_step, train_step


def _snapshot(model):
    return {k: v.clone() for k, v in model.params.items()}


def _max_delta(model, snap):
    return max((model.params[k] - v).abs().max().item() for k, v in snap.items())


def test_the_within_group_r_removes_the_prompt_confound():
    """Run 2's mechanism claim came from two group MEANS per step, which is a
    cross-step correlation: a hard prompt makes both longer completions and lower
    reward, so pooling over steps measures difficulty. Centering per group is what
    removes it, and the gate is that the two disagree on the SAME rows.
    """
    from tilerl.cli import _within_group_r

    def rows(spec):
        return [{"step": s, "g": i, "tokens": t, "reward": r}
                for s, g in enumerate(spec, 1) for i, (t, r) in enumerate(g)]

    def pooled(rs):
        mx = sum(r["tokens"] for r in rs) / len(rs)
        my = sum(r["reward"] for r in rs) / len(rs)
        den = (sum((r["tokens"] - mx) ** 2 for r in rs)
               * sum((r["reward"] - my) ** 2 for r in rs)) ** 0.5
        return sum((r["tokens"] - mx) * (r["reward"] - my) for r in rs) / den

    # Hard prompts long+low, easy short+high, but WITHIN each group the reward has
    # variance and no relation to length (which row is correct alternates).
    confounded = rows([[(1700, 0.0), (1900, 0.4)], [(1600, 0.4), (1800, 0.0)],
                       [(500, 0.8), (700, 1.0)], [(400, 1.0), (600, 0.8)]])
    assert pooled(confounded) < -0.8, "fixture must carry a strong cross-step correlation"
    assert abs(_within_group_r(confounded)) < 1e-9, (
        f"prompt difficulty survived centering: r={_within_group_r(confounded)}")

    # And a real within-group effect must still register, in both directions.
    assert _within_group_r(rows([[(400, 1.0), (1900, 0.0)], [(500, 1.0), (1800, 0.0)],
                                 [(600, 1.0), (1700, 0.0)]])) < -0.9
    assert _within_group_r(rows([[(400, 0.0), (1900, 1.0)],
                                 [(500, 0.0), (1800, 1.0)]])) > 0.9

    # No variance to correlate: None rather than a fabricated 0.0. The tied case is
    # 19 of run 2's 45 steps, and at the cap every length is identical too.
    assert _within_group_r(rows([[(400, 1.0), (1900, 1.0)]])) is None
    assert _within_group_r(rows([[(2048, 0.0), (2048, 1.0)]])) is None
    assert _within_group_r([]) is None


def test_grpo_loop_records_one_row_per_rollout():
    """The yielded tuple carries means, so a length-vs-reward claim needs the rows."""
    cfg, model = _build_model("tiny", seed=0)
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop

    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    rollouts: list = []
    steps, group = 2, 2
    # The reward must DIFFER within a group or every advantage is 0.0 and a
    # misaligned pairing is unobservable: tiny never emits EOS, so `len(c)` is
    # max_new_tokens for every rollout and the group ties. Key on the first token
    # instead, which the per-rollout seed varies.
    list(grpo_loop(engine, model, [[1, 2, 3, 4]], lambda p, c: float(c[0]), steps, backend,
                   AdamW(lr=0.0), group=group, sampling=SamplingParams(max_new_tokens=4),
                   per_rollout=rollouts))
    assert len(rollouts) == steps * group, f"{len(rollouts)} rows for {steps}x{group} rollouts"
    assert [r["step"] for r in rollouts] == [1, 1, 2, 2]
    assert sorted({r["g"] for r in rollouts}) == list(range(group))
    assert all(r["tokens"] == 4 for r in rollouts), rollouts
    # The advantage must be the one THIS row's reward earned, not another row's:
    # a misaligned zip would still write the right count of rows with the right
    # step and g, and every assertion above would pass.
    seen_signal = False
    for s in (1, 2):
        rows = [r for r in rollouts if r["step"] == s]
        want = group_advantages([r["reward"] for r in rows], group)
        assert np.allclose([r["advantage"] for r in rows], want), rows
        seen_signal = seen_signal or np.abs(want).max() > 1e-8
    assert seen_signal, (
        f"every group tied, so the advantage assertion compared zeros and could not "
        f"catch a misaligned pairing: {rollouts}")


def test_group_advantages():
    # Group-normalized: mean 0, unit std, and no signal when the group ties.
    adv = group_advantages([1.0, 2.0, 3.0, 4.0, 5.0, 5.0], group=3)
    assert abs(adv[:3].mean()) < 1e-12 and abs(adv[:3].std() - 1.0) < 1e-6
    assert np.allclose(group_advantages([2.0, 2.0], group=2), 0.0)


def test_an_all_wrong_group_is_silent_not_length_ranked():
    """Every match is 0: the length term was the whole reward, and the normalized
    advantage ranked shortest-first -- lam cancels, so no lambda could tune it and
    only lam=0 turned it off. That gradient was the 2026-09-10 P1 collapse (length
    term 1.3x the correctness term at step 1, 2.1x+ after): a group with no
    correctness signal carries no information, so its advantages are zero."""
    from tilerl.cli import _length_aware

    class Tok:
        def decode(self, ids):
            return "x" * len(ids)

    lengths = [3, 7, 1, 9, 5, 2, 8, 4]

    def adv_at(lam):
        reward = _length_aware(lambda text, gold: 0.0, {(): ""}, Tok(), lam, 32)
        return group_advantages([reward((), [0] * L) for L in lengths], group=8,
                                signal=[0.0] * 8)

    assert np.allclose(adv_at(0.1), 0.0)
    assert np.allclose(adv_at(1.0), 0.0)  # lam was never the dial; silence is
    # Without the signal the function still reports what the rewards say: the
    # zeroing is the caller's choice, not a property of the rewards.
    reward = _length_aware(lambda text, gold: 0.0, {(): ""}, Tok(), 0.1, 32)
    assert not np.allclose(
        group_advantages([reward((), [0] * L) for L in lengths], group=8), 0.0)


def test_a_constant_signal_group_yields_zero_but_a_mixed_group_judges():
    """The zeroing's control pair: groups whose non-length signal is constant go
    silent while a mixed group in the same call still judges with the right
    direction. Without the third assertion, a bug that zeroed EVERY group would
    pass -- training would quietly learn nothing (2026-09-10, the shape that bit
    four times that night)."""
    lengths = [3, 7, 1, 9, 5, 2, 8, 4]
    # All correct, lengths vary: without the rule, length ranks the advantages.
    rewards = [1.0 - 0.1 * L for L in lengths]
    assert np.allclose(group_advantages(rewards, 8, signal=[1.0] * 8), 0.0)
    # All wrong, lengths vary: same silence.
    assert np.allclose(group_advantages([-0.1 * L for L in lengths], 8,
                                        signal=[0.0] * 8), 0.0)
    # Control: a mixed group still judges -- correct rows on top.
    mixed = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    adv = group_advantages(mixed, 8, signal=mixed)
    assert not np.allclose(adv, 0.0)
    assert (adv[np.array(mixed) == 1.0] > 0).all(), adv
    assert (adv[np.array(mixed) == 0.0] < 0).all(), adv
    # A failed rollout is not a wrong answer: 7 live-correct + 1 failed is constant
    # over the live rows, so the live rows go silent too.
    live = [True] * 7 + [False]
    assert np.allclose(group_advantages([1.0] * 7 + [0.0], 8, live=live,
                                        signal=[1.0] * 8), 0.0)


def test_a_judge_that_separates_an_all_pass_group_keeps_its_signal():
    """Stage 4(b) regression: in an all-pass group the binary signal is constant,
    but the judge scores ARE the non-length signal. A judge that separates the
    rollouts must keep its ordering -- the zeroing targets length-only gradients,
    not judge-provided ones. And a judge that cannot separate stays silent."""
    judge_scores = [1.0, 0.5, 0.0, -0.5, 1.0, 0.5, 0.0, -0.5]
    adv = group_advantages(judge_scores, 8, signal=judge_scores)
    assert not np.allclose(adv, 0.0), "a separating judge was zeroed"
    assert adv[0] > adv[2], adv  # the judge's ordering survives
    # Judge cannot separate: constant signal, zeroed -- the B-hole case (the std
    # gate does not catch it: length keeps the reward spread non-zero).
    assert np.allclose(group_advantages([1.0 - 0.1 * L for L in [3, 7, 1, 9, 5, 2, 8, 4]],
                                        8, signal=[0.5] * 8), 0.0)


def test_an_empty_rollout_injects_no_gradient():
    """A zero-length completion scores 0 positions in `rl_step`, so its own
    advantage reaches nothing -- but as a reward=0 group member it moved every
    other row. The worst case is the most common one: 8-of-8 correct should be
    tied and silent, and one empty row made all seven live rows +0.378."""
    live = [True] * 7 + [False]
    assert np.allclose(group_advantages([1.0] * 7 + [0.0], 8, live=live), 0.0)
    # Without the mask, that is the bug this test exists for.
    assert not np.allclose(group_advantages([1.0] * 7 + [0.0], 8), 0.0)
    # Real signal among the live rows survives, and the dead row stays at 0.
    adv = group_advantages([1.0] * 4 + [0.0] * 3 + [0.0], 8, live=live)
    assert adv[7] == 0.0 and (adv[:4] > 0).all() and (adv[4:7] < 0).all(), adv
    # Every row dead, and one live row: no mean, no std, nothing to divide by.
    assert np.allclose(group_advantages([0.0] * 4, 4, live=[False] * 4), 0.0)
    assert np.allclose(group_advantages([1.0, 0.0, 0.0, 0.0], 4,
                                        live=[True, False, False, False]), 0.0)


def test_rl_step_matches_sft_at_unit_advantage():
    """A=1 on every row with no prompt is exactly the SFT gradient: the RL step
    must not be a second training path, only a reweighting of this one."""
    ids = np.arange(1, 2 * 12 + 1, dtype=np.int64).reshape(2, 12)
    backend = RefBackend()
    out = {}
    for name, fn in (
        ("sft", lambda m, o: train_step(m, ids, backend, o)),
        ("rl", lambda m, o: rl_step(m, ids, np.ones(2), np.ones(2, dtype=np.int64),
                                    backend, o)),
    ):
        _, model = _build_model("tiny", seed=0, keep_master=True)
        snap = _snapshot(model)
        fn(model, AdamW(lr=1e-3))
        out[name] = {k: model.params[k] - v for k, v in snap.items()}
    worst = max((out["sft"][k] - out["rl"][k]).abs().max().item() for k in out["sft"])
    assert worst < 1e-6, f"rl_step diverges from train_step at A=1: {worst:.2e}"


def test_rl_step_zero_advantage_is_a_noop():
    ids = np.arange(1, 2 * 12 + 1, dtype=np.int64).reshape(2, 12)
    _, model = _build_model("tiny", seed=0, keep_master=True)
    snap = _snapshot(model)
    rl_step(model, ids, np.zeros(2), np.full(2, 4, dtype=np.int64), RefBackend(),
            AdamW(lr=1e-3))
    assert _max_delta(model, snap) == 0.0


def test_rl_step_ignores_padding():
    """Right-padding past seq_len must not reach the gradient. Only padding:
    prompt tokens legitimately move it, through the forward pass that conditions
    every scored position on them."""
    base = np.arange(1, 12 + 1, dtype=np.int64).reshape(1, 12)
    other = base.copy()
    other[0, 9:] = 77  # padding past seq_len; causal, so no scored logit sees it
    deltas = []
    for ids in (base, other):
        _, model = _build_model("tiny", seed=0, keep_master=True)
        snap = _snapshot(model)
        rl_step(model, ids, np.array([1.0]), np.array([3]), RefBackend(),
                AdamW(lr=1e-3), seq_lens=np.array([9]))
        deltas.append({k: model.params[k] - v for k, v in snap.items()})
    worst = max((deltas[0][k] - deltas[1][k]).abs().max().item() for k in deltas[0])
    assert worst < 1e-6, f"prompt/padding leaked into the RL gradient: {worst:.2e}"


def test_grpo_length_buckets_preserve_real_token_loss_and_gradients(monkeypatch):
    import platform
    from types import SimpleNamespace

    from tilerl import train
    from tilerl.engine import SamplingParams
    from tilerl.kv_cache import NoPrefixStore

    prompt = [1, 2, 3]
    # (cap, longest, expected width). The 1200 arm is the one that catches an
    # unclamped bucket: 1200 is not a power of two, so rounding up gives 2048 --
    # 848 tokens of padding past a cap the completions can never exceed. At 2048
    # the defect is invisible, because the round-up lands on the cap itself.
    for cap, longest, width in ((2048, 300, 512), (2048, 1100, 2048), (1200, 1200, 1200)):
        comps = {i: (np.arange(n) % 100 + 4).tolist()
                 for i, n in enumerate((longest, longest // 2))}
        requests = iter(comps)
        engine = SimpleNamespace(
            # usable_slots, not a getattr default in the guard: `_require_group_fits`
            # refuses an engine narrower than the group, and a default would make it
            # pass for every stub -- the capability-check trap `_require_on_policy`
            # documents. 2 is what this stub stands in for, one slot per rollout.
            _decode_graph_on=False, _prefix=NoPrefixStore(), usable_slots=2,
            submit=lambda p, s: next(requests), step=lambda: None, poll=lambda: comps,
        )
        captured = []
        with monkeypatch.context() as m:
            m.setattr(train, "rl_step", lambda *a, **kw: captured.append((a, kw)) or 0.0)
            list(train.grpo_loop(engine, None, [prompt], lambda p, c: len(c), 1,
                                 RefBackend(), group=2,
                                 sampling=SamplingParams(max_new_tokens=cap)))
        args, kwargs = captured[0]
        batch, adv, plens = args[1:4]
        assert batch.shape == (2, len(prompt) + width), "wrong completion bucket width"
        slens = kwargs["seq_lens"]
        np.testing.assert_array_equal(slens, [len(prompt) + len(c) for c in comps.values()])
        full = np.pad(batch, ((0, 0), (0, cap - width)))
        results = []
        for ids in (batch, full):
            losses, gradients = [], {}

            class Backend(RefBackend):
                def cross_entropy_loss_grad(self, logits, tokens):
                    for row, end in enumerate(slens):
                        start = len(prompt) - 1
                        loss, _ = RefBackend().cross_entropy_loss_grad(
                            logits[row:row + 1, start:end], tokens[row:row + 1, start:end])
                        losses.append(loss)
                    return RefBackend().cross_entropy_loss_grad(logits, tokens)

            _, model = _build_model("tiny", seed=0, keep_master=True)
            clip = train.clip_grad_norm

            def capture(grads, *a):
                gradients.update({k: grads[id(v)].clone() for k, v in model.params.items()
                                  if id(v) in grads})
                return clip(grads, *a)

            with monkeypatch.context() as m:
                m.setattr(train, "clip_grad_norm", capture)
                rl_step(model, ids, adv, plens, Backend(), AdamW(lr=0), seq_lens=slens)
            assert gradients and max(g.abs().max().item() for g in gradients.values()) > 0
            results.append((losses, gradients))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
        assert results[0][1].keys() == results[1][1].keys()
        failures = []
        # atol: the conv1d gradient sums over a width-dependent padded axis; the macos-14
        # runner (3 threads, torch 2.13) measured |d| 2.0e-6 on |ref| 0.055 between widths
        # 512 and 2048, where 3 Linux/Mac hosts gave 0. Five times that, not a bug bound.
        atol = 1e-5
        for key, actual in results[0][1].items():
            ref = results[1][1][key]
            if not torch.allclose(actual, ref, rtol=1e-5, atol=atol):
                failures.append(f"{key}: max |d|={(actual - ref).abs().max().item():.9g}, "
                                f"max |ref|={ref.abs().max().item():.9g}, rtol=1e-5, atol={atol}")
        assert not failures, (
            f"cap={cap}, longest={longest}, width={width}; torch={torch.__version__}, "
            f"threads={torch.get_num_threads()}, platform={platform.platform()}\n"
            + "\n".join(failures))


def test_micro_batching_is_the_same_update():
    """A group split into micro-batches must land on the same weights as the
    whole group in one backward, or micro-batching would be a quieter way of
    shrinking the group — the thing that changes the training signal. Rows have
    different advantages, prompt lengths and seq lens, so a normalizer that
    silently follows the micro-batch size would show up here.
    """
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 300, size=(8, 20)).astype(np.int64)
    adv = np.array([1.5, -0.5, 0.0, 2.0, -1.25, 0.75, -2.0, 0.25])
    plens = np.array([3, 5, 4, 6, 3, 7, 5, 4], dtype=np.int64)
    slens = np.array([20, 18, 15, 20, 12, 19, 17, 20], dtype=np.int64)
    backend = RefBackend()

    deltas = []
    for micro in (0, 1, 3):
        _, model = _build_model("tiny", seed=0, keep_master=True)
        snap = _snapshot(model)
        rl_step(model, ids, adv, plens, backend, AdamW(lr=1e-3), seq_lens=slens, micro=micro)
        deltas.append({k: model.params[k] - v for k, v in snap.items()})
    moved = max(v.abs().max().item() for v in deltas[0].values())
    assert moved > 1e-5, "the reference step did not move the weights"
    for micro, d in zip((1, 3), deltas[1:]):
        worst = max((deltas[0][k] - d[k]).abs().max().item() for k in d)
        assert worst < 1e-6, f"micro={micro} changed the update: {worst:.2e} (moved {moved:.2e})"


def test_grpo_loop_raises_reward():
    """End to end on the tiny model: rollouts through the engine, a reward the
    policy can move, and reward must go up. The engine that samples is the model
    that trains — no second copy of the weights."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop

    torch.manual_seed(0)
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                          decode_graph=False, prefix_store=NoPrefixStore())
    half = cfg.vocab_size // 2

    # Dense reward: an untrained policy's group needs variance for a gradient at step 0.
    def reward(prompt, completion):
        return sum(1 for t in completion if t < half) / max(len(completion), 1)

    prompts = [[1, 2, 3, 4]]
    hist = list(grpo_loop(engine, model, prompts, reward, 12, backend,
                          AdamW(lr=0.05), group=6, sampling=SamplingParams(max_new_tokens=6),
                          seed=0))
    first = np.mean([r for r, *_ in hist[:3]])
    last = np.mean([r for r, *_ in hist[-3:]])
    assert last > first, f"GRPO did not raise reward: {first:.3f} -> {last:.3f}"
    # Length is logged because tied_group_fraction cannot report a bad --judge:
    # the judge reorders inside the all-pass subgroup, so it drives ties toward 0
    # whether or not it ranks anything real. A step yields a real token count.
    toks = [h[4] for h in hist]
    assert all(0 < t <= 6 for t in toks), f"completion length not reported: {toks}"


def test_grpo_loop_reports_a_step_before_the_run_ends():
    """A 100-step run that prints only on return says nothing for two hours, so a
    live run cannot be told from a hung one. The first step must be readable
    while later steps are still to come."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    gen = grpo_loop(engine, model, [[1, 2, 3, 4]], lambda p, c: float(len(c)), 3, backend,
                    AdamW(lr=1e-3), group=2, sampling=SamplingParams(max_new_tokens=4))
    first = next(gen)
    assert len(first) == 8, first  # reward, ce, secs, tied, mean tokens, timings, bucket width, tied_correctness
    assert sum(1 for _ in gen) == 2, "every step must be yielded, not just the first"


def test_an_all_correct_group_is_tied_once_more():
    """An all-correct group's rewards vary only by the length term, so its
    non-length signal is constant and the group falls silent: tied reads 1.0.

    The length term once made rewards continuous, so `tied` could never see an
    all-correct group and `tied_correctness` was added to measure it directly.
    Constant-signal zeroing makes the two coincide; tied_correctness stays
    because it measures the fact instead of inferring it from advantage sparsity.
    """
    from tilerl.train import group_advantages

    rewards = np.array([1.0 - 0.1 * L / 6144 for L in [100, 200, 300, 400, 500, 600, 700, 800]])
    signal = np.ones(8)  # corr = reward + 0.1*L/6144 = 1.0 for every row
    adv = group_advantages(rewards, 8, signal=signal)
    tied = float((adv.reshape(-1, 8) == 0).all(axis=1).mean())
    assert tied == 1.0, f"a constant-signal group must be silent, got {tied}"
    corr = np.ones(8)
    tied_correctness = float((corr.reshape(-1, 8) == corr.reshape(-1, 8)[:, :1]).all(axis=1).mean())
    assert tied_correctness == 1.0


def test_grpo_loop_yields_tied_correctness():
    """grpo_loop yields tied_correctness as the 8th element when correctness_fn is given."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    gen = grpo_loop(engine, model, [[1, 2, 3, 4]], lambda p, c: float(len(c)), 1, backend,
                    AdamW(lr=1e-3), group=2, sampling=SamplingParams(max_new_tokens=4),
                    correctness_fn=lambda p, c: 1.0)
    first = next(gen)
    assert len(first) == 8
    assert first[7] == 1.0, f"all-correct group should have tied_correctness=1.0, got {first[7]}"


def test_grpo_rollouts_are_drawn_untruncated():
    """The gradient is taken under the full softmax, so the rollout must be drawn
    from it too. A truncated or tempered sampler passed in is overridden, and the
    engine has to see the override -- checking the returned params alone would
    pass even if grpo_loop kept sampling from the caller's values."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import grpo_loop, untruncated

    card = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=4)
    assert untruncated(card) == replace(card, temperature=1.0, top_p=1.0, top_k=0)
    # max_new_tokens and the stop set are the caller's; only the measure is ours.
    keep = SamplingParams(max_new_tokens=7, stop_token_ids=(3,), top_k=20)
    assert untruncated(keep).max_new_tokens == 7
    assert untruncated(keep).stop_token_ids == (3,)

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=128, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    seen = []
    real = engine.submit
    engine.submit = lambda p, s: (seen.append(s), real(p, s))[1]
    # list() runs the generator; without it nothing submits and `seen` is empty,
    # which reads as the sampler's failure rather than the loop never starting.
    list(grpo_loop(engine, model, [[1, 2, 3, 4]], lambda p, c: float(len(c)), 1, backend,
                   AdamW(lr=1e-3), group=2, sampling=card))
    assert seen, "grpo_loop never submitted"
    for s in seen:
        assert (s.temperature, s.top_p, s.top_k) == (1.0, 1.0, 0), s


def test_last_number():
    assert last_number("so the answer is 1,234.5 dollars") == 1234.5
    assert last_number("#### -7") == -7.0
    assert last_number("no digits") is None and last_number(None) is None


def test_opd_keeps_adapter_tensor_identity():
    """A captured decode graph holds the adapter tensor objects. The
    teacher/student swap in opd_loop must copy into them, never rebind — a
    rebind samples from the captured (stale) tensors on CUDA and never raises."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.model import add_lora
    from tilerl.train import opd_loop

    cfg, model = _build_model("tiny", seed=0)
    backend = RefBackend()
    engine = build_engine(cfg, model, backend, num_blocks=64, num_slots=4,
                          decode_graph=False, prefix_store=NoPrefixStore())
    trainable = add_lora(model, rank=4)
    ids = {k: id(model.params[k]) for k in trainable}
    opd_loop(engine, model, [[1, 2, 3, 4]], 2, backend, AdamW(lr=1e-3),
             trainable=trainable, sampling=SamplingParams(max_new_tokens=4))
    assert {k: id(model.params[k]) for k in trainable} == ids


if __name__ == "__main__":  # runnable check
    test_group_advantages()
    test_rl_step_matches_sft_at_unit_advantage()
    print("rl: advantage + step OK")


def test_the_train_pool_holds_what_max_new_tokens_asks_for(tmp_path, monkeypatch):
    """A rollout cannot outgrow the pool without the engine raising, so the pool has
    to follow --max-new-tokens. A flat 512 blocks is 8192 tokens over 8 slots, 1024
    each, and past that the rollout dies on "PagedKvPool exhausted" rather than
    truncating. Assert on the kwargs build_engine is HANDED -- recomputing the
    formula here would pass either way."""
    import contextlib
    import json

    from tilerl import cli
    from tilerl import engine as engine_mod
    from tilerl.kv_cache import BLOCK_TOKENS

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": "2+2?", "answer": "4"}) + "\n")
    seen, real = {}, engine_mod.build_engine
    monkeypatch.setattr(engine_mod, "build_engine",
                        lambda *a, **kw: (seen.update(kw), real(*a, **kw))[1])
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    want = 4096
    with contextlib.suppress(SystemExit):  # a failed gate exits; the kwargs are what matter
        cli.cmd_train(cli._build_parser().parse_args(
            ["train", "--rl", "--steps", "1", "--group", "2", "--lora-rank", "2",
             "--max-new-tokens", str(want), "--data", str(data)]))
    assert seen, "build_engine was never called"
    per_slot = seen["num_blocks"] * BLOCK_TOKENS / seen["num_slots"]
    assert per_slot >= want, f"{per_slot:.0f} tokens per slot cannot hold {want}"
    assert seen["max_total_tokens"] >= want
    # Sizing from the training prompts alone once shrank both under the evals, which
    # submit prompts this path never sees: MMLU reached 515 tokens against GSM8K's
    # 183 and mmlu_score died on "request (515 tokens) exceeds max_total_tokens
    # (503)". Neither may fall under what build_engine would have given by default.
    assert seen["max_total_tokens"] >= 8192, seen["max_total_tokens"]
    assert per_slot >= 1024, f"{per_slot:.0f} tokens per slot is under the old flat pool"


def test_the_run_saves_the_adapter_that_produced_its_metrics(tmp_path, monkeypatch):
    """A gsm8k_after that beats its baseline is the run's claim, and the claim is not
    checkable without the weights. The manifest declared an `artifacts` dict that the
    train path never filled, so every finished RL run was unreproducible."""
    import contextlib
    import json

    from safetensors.torch import load_file

    from tilerl import cli

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": "2+2?", "answer": "4"}) + "\n")
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    with contextlib.suppress(SystemExit):  # a failed gate exits; the artifact is what matters
        cli.cmd_train(cli._build_parser().parse_args(
            ["train", "--rl", "--steps", "1", "--group", "2", "--lora-rank", "2",
             "--max-new-tokens", "4", "--data", str(data)]))
    run = next(p for p in tmp_path.iterdir() if (p / "manifest.json").exists())
    m = json.loads((run / "manifest.json").read_text())
    assert m["artifacts"].get("adapter") == "adapter.safetensors", m["artifacts"]
    got = load_file(str(run / "adapter.safetensors"))
    assert got, "adapter file is empty"
    assert any(k.endswith((".lora_a", ".lora_b")) or "lora" in k for k in got), sorted(got)[:5]


def test_the_judge_restores_gradient_without_crossing_the_bands():
    """A binary reward stops teaching once the policy clears the task: at 86% rollout
    accuracy a group of 8 is all-correct 30% of the time, and the 256-token run measured
    72% of steps with every advantage zero. The tiebreak has to fix that WITHOUT letting
    a judged ordering lift a wrong answer over a right one -- if it ever can, the judge
    is scoring a fact it cannot see."""
    from tilerl.judge import judge_rewards
    from tilerl.train import group_advantages

    # what the run actually hits: 8 correct rollouts, one reward, no signal
    tied = [1.0] * 8
    assert not group_advantages(tied, 8).any(), "a tied group already had gradient"

    def prefers_lower(a, b):  # a stable, transitive order, both call orders agreeing
        v = "A" if a < b else "B"
        return (v, v)

    scored, _ = judge_rewards(list(range(8)), [True] * 8, prefers_lower)
    assert group_advantages(scored, 8).any(), "the judge left the group with no gradient"

    # bands: every failure stays under every pass, whatever the judge said
    mixed, _ = judge_rewards(list(range(6)), [True, True, True, False, False, False],
                             prefers_lower)
    assert min(mixed[:3]) > max(mixed[3:]), f"a failure outranked a pass: {mixed}"

    # a judge that contradicts itself across the swap must yield no order, not a guess
    flat, _ = judge_rewards(list(range(4)), [True] * 4, lambda a, b: ("A", "A"))
    assert len(set(flat)) > 1 or not group_advantages(flat, 4).any()


def test_the_gsm8k_eval_reports_the_tokens_it_spent():
    """Length is a result on this path, not bookkeeping.

    A --judge arm's claim is that completions get SHORTER, and the eval is the only
    measurement where the training cap is absent -- so a run could otherwise finish,
    write gsm8k_after, and still answer nothing about length. The count comes from
    the ids the engine emitted: re-encoding the decoded text asks a different
    question, since decode/encode is not always a round trip.
    """
    from tilerl_kernels.backend import get_backend

    from tilerl.config import tiny
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.eval import gsm8k_accuracy
    from tilerl.model import build_random
    from tilerl.tokenizer import get_tokenizer

    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=3), get_backend(), num_blocks=128,
                          num_slots=4, max_batch=4, max_total_tokens=1024, sparse_k=0)
    rows = [{"prompt": "What is 2+2?", "answer": "4"}, {"prompt": "What is 3+5?", "answer": "8"}]
    c, n, ntok = gsm8k_accuracy(engine, get_tokenizer(None), rows,
                                SamplingParams(max_new_tokens=12), concurrency=2)
    assert (n, c <= n) == (2, True)
    # tiny never emits EOS, so every completion runs to max_new_tokens: 2 x 12 exactly.
    # An equality, not a bound -- a re-encode of decoded text would drift off it.
    assert ntok == 24, f"eval token count is not the emitted ids: {ntok}"


def test_a_recapturing_engine_clears_what_the_update_invalidated():
    """The caches an optimizer step makes stale must be cleared per step.

    A captured graph replays the forward as it was traced and a cached prefix
    serves KV from the old policy -- both silently, which is why the guard refuses
    an engine carrying either. The waivers are per cache: `recapture_graph=True`
    and `clear_prefix=True`, so a caller that turned graphs off and said only
    "graphs" is still refused for the live prefix store it forgot.

    The graphs are now KEPT across the update rather than dropped: every address
    a capture baked survives an in-place optimizer step, and the one thing that
    does not -- a cached cast -- is refilled by `invalidate_weights` instead. So
    what this asserts of the graph half is that they SURVIVE and the refill ran;
    the staleness question itself is a replay question and lives on the card.

    CPU cannot gate the graph half by comparison. Capture calls
    torch.cuda.graph_pool_handle(), which raises here, and the handler flips
    _decode_graph_on to False -- so a "captured vs eager" comparison on cpu would
    compare eager to eager and pass against any implementation at all. This gates
    the state machine instead. The sm90 half is in the bench entry.
    """
    import pytest

    from tilerl.engine import build_engine
    from tilerl.train import grpo_loop

    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    run = lambda e, **kw: list(  # noqa: E731
        grpo_loop(e, model, [[1, 2, 3]], lambda p, c: 0.0, 1, RefBackend(), group=2,
                  sampling=SamplingParams(max_new_tokens=4), **kw))

    # Same engine the guard refuses (live prefix store) -- accepted only with the flag.
    from tilerl.engine import SamplingParams

    engine = build_engine(cfg, model, RefBackend(), num_blocks=64, num_slots=4,
                          sparse_k=0)  # on-policy needs a live PrefixStore to refuse/clear
    with pytest.raises(ValueError, match="on-policy"):
        run(engine)
    # One flag is not the other: this engine's graphs are already off, so waiving
    # them changes nothing and the live prefix store must still raise.
    with pytest.raises(ValueError, match="prefix"):
        run(engine, recapture_graph=True)

    # Clearing at loop ENTRY is not enough and a one-step test cannot tell the two
    # apart: it passes either way. Re-dirty the prefix between steps, so only a
    # clear that runs AFTER EVERY update leaves it empty at the end.
    from tilerl.kv_cache import BLOCK_TOKENS

    def dirty():
        engine._decode_graphs[(1, 1)] = "stale-graph"  # cpu never captures; stand-in
        if engine._prefix.stats()["entries"] == 0:
            # insert() refcounts, it does not allocate: the pool must own the block.
            engine._prefix.insert(list(range(1, BLOCK_TOKENS + 1)), [engine._kv.alloc_block()])

    dirty()
    assert engine._prefix.stats()["entries"] == 1
    for _ in grpo_loop(engine, model, [[1, 2, 3]], lambda p, c: 0.0, 2, RefBackend(),
                       group=2, sampling=SamplingParams(max_new_tokens=4),
                       recapture_graph=True, clear_prefix=True):
        assert engine._decode_graphs == {(1, 1): "stale-graph"}, (
            "the update dropped the graphs; they are kept and their casts refilled"
        )
        assert engine._prefix.stats()["entries"] == 0, "the update left KV from the old policy"
        dirty()  # the next step must clear it again, not rely on loop entry


def test_every_adapter_receives_a_gradient():
    """An adapter no forward reads is invisible: it costs two AdamW moments and
    checkpoint bytes, trains nothing, and inflates the params number a run
    publishes. `add_lora` took any 2-D parameter as a linear weight, which caught
    a quantized weight's sidecar (`.scale` is [N, K/32]) and `conv1d` [qkv, 4] --
    32 of 64 adapters on an fp4 tiny were dead. Shape does not identify a linear;
    what `_linear` resolves an adapter FOR does.

    Both precisions, because the two causes differ: the sidecars exist only under
    fp4, conv1d is dead in either. Counting gradients rather than inspecting names
    is what makes this fail for a new dead target nobody thought of.
    """
    from tilerl.autograd import RecordingBackend, Tape
    from tilerl.config import tiny
    from tilerl.model import add_lora, build_random
    from tilerl.train import _training_kv

    for fp4 in (True, False):
        cfg = replace(tiny(), fp4=fp4)
        model = build_random(cfg, seed=0, keep_master=False)
        backend = RefBackend()
        adapters = add_lora(model, rank=4)
        tape = Tape()
        with torch.no_grad(), tape:
            logits = model.forward(np.array([[1, 2, 3, 4]]), np.arange(4, dtype=np.int64),
                                   _training_kv(model, 1, 4, device=backend.device),
                                   RecordingBackend(backend))
        grads = tape.backward(torch.ones_like(logits),
                              needs={id(v) for v in adapters.values()})
        dead = sorted(k for k, v in adapters.items() if id(v) not in grads)
        assert not dead, f"fp4={fp4}: {len(dead)} adapters never receive a gradient: {dead[:4]}"


def test_the_eval_is_not_scored_at_the_rollout_cap(tmp_path, monkeypatch):
    """Scoring the policy at the training cap measures the cap, not the policy.

    `cmd_train` built ONE SamplingParams from --max-new-tokens and handed it to both
    the rollouts and `gsm8k_accuracy`, so a run trained at 256 was also evaluated at
    256: 38.4% with mean completion 238.7, against ~82.5% uncapped
    (errors/2026-09-04-the-eval-cap-measured-itself.md, documented and unfixed).

    The gate captures the max_new_tokens gsm8k_accuracy is HANDED. Asserting the flag
    parsed would pass without the wiring, and asserting on completion lengths would
    need a model long-winded enough to reach the cap -- the tiny model is not.
    """
    import contextlib
    import json

    from tilerl import cli
    from tilerl import eval as eval_mod

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": "2+2?", "answer": "4"}) + "\n")
    held = tmp_path / "held.jsonl"
    held.write_text(json.dumps({"prompt": "3+3?", "answer": "6"}) + "\n")
    seen = {}
    real = eval_mod.gsm8k_accuracy

    def spy(engine, tok, rows, sampling, **kw):
        seen["eval_cap"] = sampling.max_new_tokens
        return real(engine, tok, rows, sampling, **kw)

    # cli.py imports gsm8k_accuracy into cmd_train's LOCAL scope at call time, so
    # there is no module attribute to patch -- patch the source module instead.
    monkeypatch.setattr("tilerl.eval.gsm8k_accuracy", spy)
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    with contextlib.suppress(SystemExit):
        cli.cmd_train(cli._build_parser().parse_args(
            ["train", "--rl", "--steps", "1", "--group", "2", "--lora-rank", "2",
             "--max-new-tokens", "4", "--eval-max-new-tokens", "64",
             "--data", str(data), "--eval-gsm8k", str(held)]))
    assert seen, "gsm8k_accuracy was never called"
    assert seen["eval_cap"] == 64, (
        f"the eval ran at max_new_tokens={seen['eval_cap']}, the ROLLOUT cap; "
        "the eval measures the cap rather than the policy")


def test_a_loaded_adapter_actually_changes_the_output(tmp_path, monkeypatch):
    """A loader that attaches nothing re-scores the BASE model and returns a clean
    number under a trained run's name -- the same failure shape as the dead adapters
    in #98, where 32 tensors were carried, optimized and saved without ever being read.

    So the gate is behavioural, not structural: greedy-decode the same prompt with and
    without the loaded adapter and require the token sequences to differ. The adapter
    is trained for a few real steps rather than filled with noise, so "differs" means
    the trained weights reached the forward, not that random values broke it.
    """
    import contextlib
    import json

    from safetensors.torch import load_file

    from tilerl import cli

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": "2+2?", "answer": "4"}) + "\n")
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    # No --data, so cmd_train takes its DENSE reward (fraction of tokens below
    # vocab/2, cli.py:363) instead of exact-match on a gold answer. Exact-match on a
    # tiny random model scores 0 for every rollout, every group ties,
    # group_advantages returns zeros, and lora_b never leaves its zero init -- the
    # adapter would be an exact no-op and `differs` would fail for a reason that has
    # nothing to do with the loader. --recipe grpo-tiny-smoke is what allows no --data.
    argv = ["train", "--rl", "--recipe", "grpo-tiny-smoke", "--model", "tiny",
            "--steps", "4", "--group", "4", "--lora-rank", "2",
            "--max-new-tokens", "6", "--lr", "0.5"]
    with contextlib.suppress(SystemExit):
        cli.cmd_train(cli._build_parser().parse_args(argv))
    run = next(p for p in tmp_path.iterdir() if (p / "manifest.json").exists())
    saved = run / "adapter.safetensors"
    assert saved.exists(), "no adapter to load"
    # lora_b SPECIFICALLY: it inits to zero, so y + B(Ax) == y until it moves. A
    # nonzero lora_a proves only that add_lora ran.
    got = load_file(str(saved))
    assert any(v.abs().max() > 0 for k, v in got.items() if k.endswith(".lora_b")), (
        "every lora_b is still zero, so the adapter is an exact no-op and this test "
        "could not tell a working loader from a broken one")

    def decode(load: str | None) -> list[int]:
        from tilerl_kernels.backend import get_backend

        from tilerl.cli import _build_engine, _build_model
        from tilerl.engine import SamplingParams
        from tilerl.model import add_lora

        cfg, model = _build_model("tiny", seed=0, keep_master=False)
        engine = _build_engine(cfg, model, get_backend())
        trainable = add_lora(model, rank=2)
        if load:
            cli._load_adapter(trainable, load, lambda *a, **k: None)
        rid = engine.submit([1, 2, 3, 4], SamplingParams(max_new_tokens=8, temperature=0.0))
        for _ in range(64):
            engine.step()
            if rid in (done := engine.poll()):
                return list(done[rid])
        raise AssertionError("decode never finished")

    assert decode(None) != decode(str(saved)), (
        "the loaded adapter did not change the output: it was not attached to the "
        "tensors the forward reads, so an eval would re-score the base model")

    # And a file that does not match is refused, not silently partially applied.
    import torch
    from safetensors.torch import save_file

    bad = tmp_path / "bad.safetensors"
    good = load_file(str(saved))
    save_file({**good, "layers.0.q_proj.scale.lora_a": torch.zeros(2, 2)}, str(bad))
    with contextlib.suppress(SystemExit):
        decode(str(bad))
        raise AssertionError("an adapter with an unknown key was accepted")


def test_the_training_engine_keeps_its_decode_graph(tmp_path, monkeypatch):
    """The graph is 2.16x on the 27B RL step, and the flag that buys it is one word.

    `cmd_train` built its engine with `decode_graph=False` because a graph traced on
    old weights replays the old policy silently. #94 made that recoverable --
    `recapture_graph=True` clears the graphs after every update -- and the card
    measured what it is worth: 73.62 -> 34.09 s/step, a 100-step run 123 -> 57 min
    (wins/2026-09-05-recapture-after-update.md).

    Both halves are asserted because either alone is a silent regression: graphs on
    without the waiver raises, and the waiver without graphs on is a no-op that
    still reads as "recapture is enabled".
    """
    import contextlib
    import json

    from tilerl import cli
    from tilerl import engine as engine_mod
    from tilerl import train as train_mod

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"prompt": "2+2?", "answer": "4"}) + "\n")
    seen = {}
    real_build, real_loop = engine_mod.build_engine, train_mod.grpo_loop

    def spy_build(*a, **kw):
        seen["decode_graph"] = kw.get("decode_graph")
        return real_build(*a, **kw)

    def spy_loop(*a, **kw):
        seen["recapture_graph"] = kw.get("recapture_graph")
        return real_loop(*a, **kw)

    monkeypatch.setattr("tilerl.engine.build_engine", spy_build)
    monkeypatch.setattr("tilerl.train.grpo_loop", spy_loop)
    monkeypatch.setattr("tilerl.ledger.runs_root", lambda: tmp_path)
    with contextlib.suppress(SystemExit):
        cli.cmd_train(cli._build_parser().parse_args(
            ["train", "--rl", "--steps", "1", "--group", "2", "--lora-rank", "2",
             "--max-new-tokens", "4", "--data", str(data)]))
    assert seen.get("decode_graph") is True, (
        f"the training engine was built with decode_graph={seen.get('decode_graph')}; "
        "the RL step pays 2.16x for that")
    assert seen.get("recapture_graph") is True, (
        "grpo_loop was not told to recapture, so a kept graph would replay the "
        "weights it was traced on")


def test_an_all_right_group_is_silent_and_lambda_matters_only_in_mixed_groups():
    """The length term's stated purpose was breaking all-right ties (run 2 collapsed
    in one at step 41). That half is now void: a group whose non-length signal is
    constant carries no learnable ordering, so its advantages are zero -- a
    length-only gradient was the 2026-09-10 P1 collapse. The judge (stage 4b)
    breaks a tie when it can separate the rollouts; without it, a tie is not
    broken.

    The kept half: in a MIXED group lambda is a real dial, and the reward-level
    bound still holds. The negative control stays: a mixed group must carry
    signal, or "all zero" would mean `group_advantages` is broken rather than
    indifferent.
    """
    cap, group = 2048, 8
    lens = [412, 1893, 655, 1204, 988, 1560, 301, 2048]

    def shaped(correct, lam):
        return [(1.0 if c else 0.0) - lam * (n / cap) for c, n in zip(correct, lens)]

    # today's behaviour: every rollout right, no length term -> no gradient at all
    assert np.allclose(group_advantages([1.0] * group, group), 0.0)
    # negative control: correctness varies, so the function does produce signal
    mixed = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert np.abs(group_advantages(mixed, group)).min() > 0.5, (
        "a mixed group must carry signal, or the all-right zeros above prove nothing")

    # with the term, an all-right group is STILL silent: the signal is constant, so
    # the length term no longer decides the gradient -- for every lambda. It was
    # never a dial here: the ratio -(L-Lbar)/std(L) has no lam in it.
    for lam in (1e-4, 1e-2, 0.1, 0.5, 1.0):
        assert np.allclose(group_advantages(shaped([True] * group, lam), group,
                                            signal=[1.0] * group), 0.0), lam

    # but in a MIXED group lambda is a real dial -- so it is bounded, not free
    lo = group_advantages(shaped(mixed_c := [True] * 4 + [False] * 4, 1e-4), group)
    hi = group_advantages(shaped(mixed_c, 1.0), group)
    assert np.abs(hi - lo).max() > 0.1, "lam must matter where correctness varies"

    # the bound: at lam <= cap/(cap-1) a short wrong answer never outranks a long right one
    worst = [cap] + [1] * (group - 1)
    for lam, inverted in ((1.0, False), (1.001, True)):
        r = [1.0 - lam * (worst[0] / cap)] + [-lam * (n / cap) for n in worst[1:]]
        assert (max(r[1:]) > r[0]) is inverted, f"lam={lam} inversion != {inverted}"

    # the case a length term provably cannot reach: every rollout at the cap
    for lam in (0.1, 1.0):
        at_cap = [(1.0 if c else 0.0) - lam * (cap / cap) for c in [True] * group]
        assert np.allclose(group_advantages(at_cap, group), 0.0), (
            "steps 41 and 44 sat exactly at the cap; any function of length is constant "
            "within the group there and only a bigger cap recovers them")


def test_the_rl_reward_closure_carries_the_length_term_and_the_matcher_does_not():
    """The term belongs in the RL reward and NOT in `MATCHERS`: the same matcher feeds
    `gsm8k_accuracy`, whose count becomes `manifest["metrics"]["gsm8k_*"]` -- the number P1's
    exit criterion reads. This drives the real closure rather than a copy of its arithmetic.
    """
    from tilerl.cli import _length_aware
    from tilerl.eval import MATCHERS

    class Tok:
        def decode(self, ids):
            return "the answer is 42" if 42 in ids else "the answer is 7"

    prompt, cap = (1, 2, 3), 100
    gold = {prompt: "42"}
    short, long_ = [42], [42] + [9] * 79        # same answer, 80x the tokens

    r = _length_aware(MATCHERS["number"], gold, Tok(), 0.1, cap)
    assert r(prompt, short) > r(prompt, long_), "the length term is not reaching the reward"
    assert abs((r(prompt, short) - r(prompt, long_)) - 0.1 * 79 / cap) < 1e-12

    # lam=0 reproduces the old behaviour exactly, so the flag can disable the term
    off = _length_aware(MATCHERS["number"], gold, Tok(), 0.0, cap)
    assert off(prompt, short) == off(prompt, long_) == 1.0

    # correctness still dominates: a short wrong answer stays below a long right one
    assert r(prompt, long_) > r(prompt, [7])

    # the matcher is untouched -- it is what the P1 gate counts through
    assert MATCHERS["number"]("the answer is 42", "42") is True



def test_prompts_per_step_normalises_each_prompt_separately():
    """A step of two prompts is two independent groups, not one group of 2*g.

    `reshape(-1, group)` infers the group count from the length, so it accepts
    16 rewards at group=8 as (2, 8) whether the caller meant two prompts or one
    prompt whose rollout count doubled -- and normalising two prompts as one
    prompt's halves degrades GRPO to REINFORCE with no raise. `groups` is the
    caller's own expectation, which is the only thing that can tell them apart.
    """
    import pytest

    from tilerl.train import group_advantages

    # Two prompts of 2: the first disagrees, the second is tied and must stay silent.
    # A single group of 4 over the same rewards would give all four rows a nonzero
    # advantage -- which is the bug, so this asserts the shape and not just non-zero.
    adv = group_advantages([1.0, 0.0, 5.0, 5.0], group=2, groups=2)
    assert np.allclose(adv, [1.0, -1.0, 0.0, 0.0])
    flat = group_advantages([1.0, 0.0, 5.0, 5.0], group=4, groups=1)
    assert not np.allclose(flat[2:], 0.0), "the one-group reading must differ, or the test is blind"

    # The wrong group count raises instead of reshaping into it.
    with pytest.raises(ValueError, match="averaging across prompts"):
        group_advantages([1.0] * 16, group=8, groups=1)
    # And a partial group still raises, which reshape would also have caught.
    with pytest.raises(ValueError, match="multiple of the group"):
        group_advantages([1.0] * 7, group=2)
    # Default groups=None keeps every existing caller working.
    assert np.allclose(group_advantages([2.0, 2.0], group=2), 0.0)


def test_a_step_of_two_prompts_does_not_normalise_across_them(tmp_path):
    """End to end: two prompts in one step, each centred within its own group.

    The failure this guards is silent. With a per-prompt reward offset (prompt 1
    scores ~10 higher because it is easier), normalising the step as ONE group of
    4 gives BOTH of prompt 1's rows a positive advantage -- gradient for being the
    easy prompt, which is prompt difficulty leaking in. Measured on this fixture:
    correct [+1, -1, +1, -1] against the bug's [-0.896, -1.095, +1.095, +0.896].
    So the two readings differ in SIGN on two rows, not just in magnitude.
    """
    from tilerl_kernels.backend import get_backend

    import tilerl.train as train_mod
    from tilerl.cli import _build_model
    from tilerl.engine import BLOCK_TOKENS, SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore

    cfg, model = _build_model("tiny", seed=0)
    backend = get_backend()
    # Different lengths on purpose: a scalar `plens` (np.full(group, len(prompt)))
    # would mis-mask every row of the shorter prompt.
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]
    seen: dict = {}

    def reward(p, c):
        k = tuple(int(x) for x in p)
        seen[k] = seen.get(k, 0) + 1
        return float(seen[k] % 2) + (10.0 if len(k) > 3 else 0.0)

    rows = 4
    ctx = max(max(map(len, prompts)) + 8 + 64, 128)
    engine = build_engine(cfg, model, backend, num_slots=rows, max_batch=rows,
                         num_blocks=-(-ctx // BLOCK_TOKENS) * rows + 8,
                         max_total_tokens=max(ctx, 512), decode_graph=False,
                         prefix_store=NoPrefixStore())
    per: list = []
    issued: list = []
    real_submit = engine.submit

    def spy(input_ids, params=None):
        issued.append(None if params is None else params.seed)
        return real_submit(input_ids, params)

    engine.submit = spy
    handed: list = []
    real_rl_step = train_mod.rl_step

    def rl_spy(model_, batch, adv, plens, *a, **kw):
        handed.append((list(map(int, plens)), list(map(int, kw["seq_lens"])), batch))
        return real_rl_step(model_, batch, adv, plens, *a, **kw)

    monkeypatch_target = train_mod
    monkeypatch_target.rl_step = rl_spy
    # Two steps, not one: a stride bug is invisible within a single step, since the
    # step index multiplies it.
    steps = list(train_mod.grpo_loop(
        engine, model, prompts, reward, 2, backend, group=2, prompts_per_step=2,
        sampling=SamplingParams(max_new_tokens=6), seed=0, per_rollout=per))

    assert len(per) == 2 * rows, f"two steps of prompts_per_step * group rows, got {len(per)}"
    assert sorted({r["p"] for r in per}) == [0, 1], "both prompts must appear"
    # Each group centred on itself: every prompt's advantages sum to ~0 SEPARATELY.
    # The whole step summing to 0 is necessary but not sufficient -- the buggy
    # reading above also sums to 0 across the step.
    for p in (0, 1):
        adv = [r["advantage"] for r in per if r["p"] == p and r["step"] == 1]
        assert abs(sum(adv)) < 1e-9, f"prompt {p} advantages are not centred: {adv}"
        assert len(adv) == 2 and adv[0] * adv[1] < 0, (
            f"prompt {p} must have one positive and one negative row, got {adv}")
    # And the step reported a real tie fraction, not a constant.
    assert steps[0][3] == 0.0, f"neither group ties on this fixture, got {steps[0][3]}"
    assert len(issued) == 2 * rows, f"expected {2 * rows} submits, saw {len(issued)}"
    train_mod.rl_step = real_rl_step
    assert len(handed) == 2, f"one rl_step per step, saw {len(handed)}"

    # The other two silent failures on this path, which the advantage assertions above
    # do NOT cover -- both were mutation-tested and survived them.
    #
    # (1) The seed must stride by prompts_per_step * group. Striding by `group`, as the
    #     one-prompt code did, reissues step N-1's seeds to every prompt after the first:
    #     prompt 1 of step 0 would draw the same completions as prompt 0 of step 1.
    #     Read off the engine's OWN submits, not recomputed from train.py's formula --
    #     a test that mirrors the expression it guards passes whatever the expression
    #     becomes.
    assert len(set(issued)) == len(issued), f"a seed repeats within two steps: {issued}"

    # (2) Every row's prompt length is its OWN. A scalar plens (np.full(rows,
    #     len(picks[0]))) masks prompt 1's rows against prompt 0's length -- 3 vs 6 here,
    #     so three prompt positions of every prompt-1 row enter the scored span and three
    #     of its completion positions leave it. Caught by reading what rl_step was handed:
    #     a wrong mask changes the gradient, not the shape, so nothing downstream raises.
    assert len(prompts[0]) != len(prompts[1]), (
        "the fixture needs unequal prompt lengths, or (2) cannot fail")
    for plens, slens, batch in handed:
        assert sorted(set(plens)) == sorted({len(p) for p in prompts}), (
            f"plens must carry each row's own prompt length, got {sorted(set(plens))} "
            f"against prompts of {[len(p) for p in prompts]}")
        # And the scored span is exactly the completion: every row's prompt sits below
        # plens and its padding above slens, with nothing of one inside the other.
        for row, (pl, sl) in enumerate(zip(plens, slens)):
            assert 0 < pl <= sl <= batch.shape[1], f"row {row}: {pl} .. {sl}"


def test_clearing_the_prefix_after_update_keeps_rollouts_token_for_token_eager(monkeypatch):
    """On-policy after an update with a live PrefixStore: the second step reuses the first
    prompt's cached KV. clear_prefix=True (engine.invalidate_weights after every update)
    must make those completions token-for-token identical to a store-free (always-eager)
    engine under the same seeds; a loop that SKIPS the clear serves the old policy's KV and
    must diverge.

    CPU can only gate the prefix half -- the graph half needs cuda graph capture (a CPU
    graph replays eager, so a graph comparison here would pass against anything); the
    graph token-equality and the within-5% wall clock are pending-remote on sm90:

        TILERL_TARGET=cuda python scripts/<recapture bench>   # graph == eager, >=0.95 speed
    """
    import tilerl.train as train_mod
    from tilerl.autograd import AdamW
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import BLOCK_TOKENS, NoPrefixStore

    # Step 1 caches a 256-token prefix (16 blocks); step 2 submits a 336-token extension so
    # it takes a genuine PARTIAL hit (a full-length hit is treated as a miss and recomputed).
    base = list(np.random.RandomState(7).randint(3, 320, size=16 * BLOCK_TOKENS))
    longer = base + list(np.random.RandomState(8).randint(3, 320, size=5 * BLOCK_TOKENS))
    prompts = [base, longer]

    # Content (not length) reward: group members that sampled different tokens must get
    # different advantages or the step carries zero gradient and no update moves weights.
    def reward(prompt, completion):
        return float(sum(int(t) % 7 for t in completion))

    # Capture completions per step without changing the loop under test.
    real_drain = train_mod._drain
    captured = {}

    def spy_drain(engine, ids, what):
        done = real_drain(engine, ids, what)
        captured[len(captured)] = [tuple(done[i]) for i in ids]
        return done

    monkeypatch.setattr(train_mod, "_drain", spy_drain)
    monkeypatch.setattr(train_mod, "_require_on_policy", lambda *a, **k: None)

    def run(*, no_store, clear_prefix):
        captured.clear()
        torch.manual_seed(123)
        cfg, model = _build_model("tiny", seed=0, keep_master=True)
        engine = build_engine(
            cfg, model, RefBackend(), num_blocks=64, num_slots=4, max_batch=4,
            max_total_tokens=4096, sparse_k=0,  # dense: needs a real live PrefixStore arm
            prefix_store=NoPrefixStore() if no_store else None)  # default live PrefixStore
        optimizer = AdamW(lr=0.1)
        for _ in train_mod.grpo_loop(
                engine, model, prompts, reward, 2, RefBackend(), optimizer=optimizer,
                group=2, prompts_per_step=1,
                sampling=SamplingParams(max_new_tokens=6), seed=100,
                clear_prefix=clear_prefix):
            pass
        return captured[1], engine.stats()  # post-update (second-step) completions + store

    reference, _ = run(no_store=True, clear_prefix=False)  # no store: KV always recomputed
    cleared, clr_stats = run(no_store=False, clear_prefix=True)  # live store, cleared per update
    stale, hit_stats = run(no_store=False, clear_prefix=False)  # stale KV survives update

    assert cleared == reference, (
        "after an update a cleared prefix store must sample exactly what a store-free "
        "engine does; the invalidation let old-policy KV survive")
    # The clear must empty the store before step 2: the 256-token prefix published in
    # step 1 is gone, so the extension admits as a miss. A clear that fails to call
    # PrefixStore.clear() leaves a hit here AND (above) stale KV in the completions.
    assert clr_stats["prefix_hits"] == 0, clr_stats
    # Positive proof the 256/336 partial hit actually happens on the live store; without
    # it the stale-vs-reference gap below is unattributable and a _match_prefix
    # threshold change turns both live runs into eager runs that stay green.
    assert hit_stats["prefix_hits"] >= 1, hit_stats
    assert hit_stats["prefix_hit_tokens"] >= 16 * BLOCK_TOKENS, hit_stats
    assert stale != reference, (
        "skipping invalidate_weights did not change the post-update samples -- the gate "
        "cannot see a stale prefix; the partial-hit/nonzero-advantage fixture is vacuous")
