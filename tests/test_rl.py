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


def test_group_advantages():
    # Group-normalized: mean 0, unit std, and no signal when the group ties.
    adv = group_advantages([1.0, 2.0, 3.0, 4.0, 5.0, 5.0], group=3)
    assert abs(adv[:3].mean()) < 1e-12 and abs(adv[:3].std() - 1.0) < 1e-6
    assert np.allclose(group_advantages([2.0, 2.0], group=2), 0.0)


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
    assert len(first) == 5, first  # reward, ce, secs, tied, mean completion tokens
    assert sum(1 for _ in gen) == 2, "every step must be yielded, not just the first"


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
                          num_slots=4, max_batch=4, max_total_tokens=1024)
    rows = [{"prompt": "What is 2+2?", "answer": "4"}, {"prompt": "What is 3+5?", "answer": "8"}]
    c, n, ntok = gsm8k_accuracy(engine, get_tokenizer(None), rows,
                                SamplingParams(max_new_tokens=12), concurrency=2)
    assert (n, c <= n) == (2, True)
    # tiny never emits EOS, so every completion runs to max_new_tokens: 2 x 12 exactly.
    # An equality, not a bound -- a re-encode of decoded text would drift off it.
    assert ntok == 24, f"eval token count is not the emitted ids: {ntok}"
