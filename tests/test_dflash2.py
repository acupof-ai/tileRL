"""Hermetic CPU gates for the DFlash2 block drafter.

The whole block forward is compared against a torch-eager transcription of
vLLM's ``qwen3_dflash2.py`` — per-slot loops for the conv taps, an explicit
attention mask, the full [K,K] transition score and the sequential walk the
triton kernel runs. Two invariants get their own test because they are silent
when broken: the conv's tap must stop at the block start, and the walk must
follow the predecessor it just emitted.
"""

from __future__ import annotations

import json
import math
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import pytest
import torch
from safetensors.torch import save_file

from tilerl.config import tiny
from tilerl.dflash2 import load_dflash2
from tilerl.model import build_random
from tilerl.spec import _DRAFT_TOP, read_head_params
from tilerl_kernels.backend import get_backend

_NORM = 0.25
#: The tiny trunk's random readout leaves ~10 logits between its top two tokens.
#: The selector needs a comparable scale or the walk never moves the pick and the
#: parity test would not exercise it; the checkpoint's codebooks are trained to.
_SELECTOR_SCALE = 50.0
_HF = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 320,
    "rope_parameters": {"rope_theta": 10000.0},
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "sliding_window": 6,  # small enough that the window bites inside the block
    "is_causal": False,
    "dflash_config": {
        "block_size": 4,
        "conv_group_size": 8,
        "conv_kernel_size": 2,
        "mask_token_id": 319,
        "selector_rank": 6,
        "selector_top_k": 3,
        "target_layer_ids": [0, 1],
    },
}


def _tiny_head(tmp_path):
    """A random DFlash2 head on disk, loaded against a ``tiny()`` trunk."""
    g = torch.Generator().manual_seed(0)

    def rnd(*shape):
        return (torch.randn(shape, generator=g) * 0.2).to(torch.bfloat16)

    h, d, dc = _HF["hidden_size"], _HF["head_dim"], _HF["dflash_config"]
    inter, v, hq, hkv = _HF["intermediate_size"], _HF["vocab_size"], 4, 2
    taps, groups = dc["conv_kernel_size"], h // dc["conv_group_size"]
    t = {
        "fc.weight": rnd(h, len(dc["target_layer_ids"]) * h),
        "hidden_norm.weight": torch.full((h,), _NORM, dtype=torch.bfloat16),
        "norm.weight": torch.full((h,), _NORM, dtype=torch.bfloat16),
        "candidate_selector.hidden_projection.weight": rnd(dc["selector_rank"], h),
        "candidate_selector.predecessor_codebook": rnd(v, dc["selector_rank"]) * _SELECTOR_SCALE,
        "candidate_selector.successor_codebook": rnd(v, dc["selector_rank"]) * _SELECTOR_SCALE,
    }
    for i in range(_HF["num_hidden_layers"]):
        t |= {
            f"layers.{i}.input_layernorm.weight": torch.full((h,), _NORM, dtype=torch.bfloat16),
            f"layers.{i}.post_attention_layernorm.weight": torch.full(
                (h,), _NORM, dtype=torch.bfloat16
            ),
            f"layers.{i}.self_attn.q_proj.weight": rnd(hq * d, h),
            f"layers.{i}.self_attn.k_proj.weight": rnd(hkv * d, h),
            f"layers.{i}.self_attn.v_proj.weight": rnd(hkv * d, h),
            f"layers.{i}.self_attn.o_proj.weight": rnd(h, hq * d),
            f"layers.{i}.self_attn.q_norm.weight": torch.full((d,), 1.0, dtype=torch.bfloat16),
            f"layers.{i}.self_attn.k_norm.weight": torch.full((d,), 1.0, dtype=torch.bfloat16),
            f"layers.{i}.mlp.gate_proj.weight": rnd(inter, h),
            f"layers.{i}.mlp.up_proj.weight": rnd(inter, h),
            f"layers.{i}.mlp.down_proj.weight": rnd(h, inter),
            f"layers.{i}.attention_conv.base_kernel": rnd(2, taps, h),
            f"layers.{i}.attention_conv.kernel_projection.weight": rnd(2 * taps * groups, h),
            f"layers.{i}.mlp_conv.base_kernel": rnd(2, taps, h),
            f"layers.{i}.mlp_conv.kernel_projection.weight": rnd(2 * taps * groups, h),
        }
    (tmp_path / "config.json").write_text(json.dumps(_HF))
    p = tmp_path / "model.safetensors"
    save_file(t, str(p))
    return load_dflash2(build_random(tiny(), seed=0), p)


# --- torch-eager reference (vLLM qwen3_dflash2.py, transcribed) --------------
def _rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope(x, pos, theta):
    """rotate_half on [T, heads, D], the convention reference.rope uses."""
    half = x.shape[-1] // 2
    ang = pos[:, None].float() * theta ** (-torch.arange(half).float() / half)
    c, s = ang.cos()[:, None], ang.sin()[:, None]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


def _ref_conv(x, delta, base, taps, groups, gsize):
    """One tap per output slot, skipping the taps that would reach before the
    block start (vLLM's ``min(taps, position + 1)``)."""
    blocks = x.reshape(-1, groups, gsize)
    base = base.reshape(taps, groups, gsize)
    out = torch.zeros_like(blocks)
    for slot in range(blocks.shape[0]):
        for tap in range(min(taps, slot + 1)):
            out[slot] += (base[tap] + delta[slot, tap][:, None]) * blocks[slot - tap]
    return out.reshape(x.shape)


def _reference_draft(head, aux, positions, anchor):
    p, cfg, dc = head.params, head.cfg, head.dcfg
    eps, groups, gs = cfg.rms_eps, head.groups, dc.group_size
    hq, hkv, D, S = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim, dc.block_size

    def lin(x, key):
        return x.float() @ p[key].float().t()

    def conv_in(x, norm_key, conv_key):
        h = _rms(x, p[norm_key].float(), eps)
        c = lin(h, f"{conv_key}.proj").reshape(-1, 2, dc.taps, groups)
        base = p[f"{conv_key}.base"].float()
        return _ref_conv(h, c[:, 0], base[0], dc.taps, groups, gs), c[:, 1]

    hc = _rms(lin(aux, "fc"), p["hidden_norm"].float(), eps)
    ctx = []
    for i in range(cfg.num_layers):
        k = lin(hc, f"layers.{i}.k_proj").reshape(-1, hkv, D)
        k = _rope(_rms(k, p[f"layers.{i}.k_norm"].float(), eps), positions, cfg.rope_theta)
        ctx.append((k, lin(hc, f"layers.{i}.v_proj").reshape(-1, hkv, D)))

    start = int(positions[-1]) + 1
    pos = torch.arange(start, start + S)
    ids = torch.tensor([anchor] + [dc.mask_token_id] * (S - 1))
    x = head.trunk.params["embed_tokens"].float()[ids]
    mask = pos[:, None] - torch.cat([positions, pos])[None, :] >= dc.sliding_window
    for i in range(cfg.num_layers):
        pre = f"layers.{i}"
        h, c = conv_in(x, f"{pre}.input_norm", f"{pre}.attn_conv")
        q = lin(h, f"{pre}.q_proj").reshape(S, hq, D)
        q = _rope(_rms(q, p[f"{pre}.q_norm"].float(), eps), pos, cfg.rope_theta)
        k = lin(h, f"{pre}.k_proj").reshape(S, hkv, D)
        k = _rope(_rms(k, p[f"{pre}.k_norm"].float(), eps), pos, cfg.rope_theta)
        k = torch.cat([ctx[i][0], k]).repeat_interleave(hq // hkv, 1)
        v = torch.cat([ctx[i][1], lin(h, f"{pre}.v_proj").reshape(S, hkv, D)])
        att = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(D)
        att = torch.softmax(att.masked_fill(mask, float("-inf")), -1)
        o = torch.einsum("hqk,khd->qhd", att, v.repeat_interleave(hq // hkv, 1))
        o = lin(o.reshape(S, -1), f"{pre}.o_proj")
        x = x + _ref_conv(o, c, p[f"{pre}.attn_conv.base"].float()[1], dc.taps, groups, gs)
        h, c = conv_in(x, f"{pre}.post_attn_norm", f"{pre}.mlp_conv")
        m = torch.nn.functional.silu(lin(h, f"{pre}.gate_proj")) * lin(h, f"{pre}.up_proj")
        m = lin(m, f"{pre}.down_proj")
        x = x + _ref_conv(m, c, p[f"{pre}.mlp_conv.base"].float()[1], dc.taps, groups, gs)
    hidden = _rms(x, p["norm"].float(), eps)[1:]

    logits = hidden @ head.trunk.params["embed_tokens"].float().t()
    unary, cand = torch.topk(logits, dc.top_k, dim=-1)
    proj = lin(hidden, "selector.proj")
    pred, succ = p["selector.pred"].float(), p["selector.succ"].float()
    # the full [steps, K, K] transition score, then the walk over its rows
    prev_ids = torch.cat([torch.full((1, dc.top_k), anchor), cand[:-1]])
    scores = unary[:, None, :] + torch.einsum(
        "spr,scr->spc", pred[prev_ids] * proj[:, None], succ[cand]
    )
    out, row = [], 0
    for step in range(hidden.shape[0]):
        row = int(scores[step, row].argmax())
        out.append(int(cand[step, row]))
    return hidden, out


# --- gates ------------------------------------------------------------------
def test_block_draft_matches_reference(tmp_path):
    """The whole block forward, against the transcribed reference."""
    head = _tiny_head(tmp_path)
    positions = torch.arange(5)
    g = torch.Generator().manual_seed(1)
    aux = torch.randn(1, 5, 2 * _HF["hidden_size"], generator=g)

    tokens = head.draft(aux, positions, anchor=7, backend=get_backend())
    want_hidden, want_tokens = _reference_draft(head, aux[0], positions, 7)

    got_hidden = head.block_hidden(
        head.context_kv(aux, positions, get_backend()), positions, 7, 5, get_backend()
    )
    assert torch.allclose(got_hidden[0, 1:], want_hidden, rtol=1e-2, atol=1e-2)
    assert tokens == want_tokens, (tokens, want_tokens)
    assert len(tokens) == _HF["dflash_config"]["block_size"] - 1
    top1 = (want_hidden @ head.trunk.params["embed_tokens"].float().t()).argmax(-1).tolist()
    assert tokens != top1, "degenerate fixture: the selector never moved the pick"


def test_conv_tap_stops_at_the_block_start(tmp_path):
    """Slot 0 must carry only its own term.

    The two-tap conv is what lets a later slot see the earlier ones, so its
    shift reaches backwards — and at slot 0 that is the previous block, whose
    tokens this pass never drafted. vLLM lays the blocks out flat and masks by
    ``position % block_size``; leaking there is silent, the draft just degrades.
    """
    head = _tiny_head(tmp_path)
    dc, g = head.dcfg, head.groups
    x = torch.randn(1, dc.block_size, head.cfg.hidden_size)
    delta = torch.randn(1, dc.block_size, dc.taps, g)
    base = torch.randn(dc.taps, head.cfg.hidden_size)

    out = head._conv(x, delta, base)
    tap0 = (base[0] + delta[..., 0, :].repeat_interleave(dc.group_size, -1)) * x
    assert torch.allclose(out[:, 0], tap0[:, 0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(out[:, 1], tap0[:, 1], rtol=1e-2)  # tap 1 does reach slot 1


def test_selector_walk_follows_the_token_it_emitted(tmp_path):
    """Two slots, hand-built: the walk must take 12 -> 5, not the row-0 best.

    Slot 1 is scored against the token slot 0 emitted. Reading a fixed row (or
    slot 0's own best candidate) picks 9 here, and every downstream token is
    then conditioned on a predecessor that was never drafted.
    """
    head = _tiny_head(tmp_path)
    dc, rank = head.dcfg, head.dcfg.rank
    v = head.cfg.vocab_size
    head.params["selector.pred"] = torch.zeros(v, rank)
    head.params["selector.succ"] = torch.zeros(v, rank)
    # predecessor 12 sends candidate 5 to the top; the anchor's row prefers 12.
    head.params["selector.pred"][12, 0] = 1.0
    head.params["selector.succ"][5, 0] = 10.0
    head.params["selector.succ"][9, 0] = -10.0

    unary = torch.zeros(2, v)
    unary[0, 12], unary[0, 9], unary[0, 3] = 3.0, 2.0, 1.0
    unary[1, 9], unary[1, 5], unary[1, 3] = 3.0, 2.0, 1.0
    head.trunk = _StubTrunk(unary)
    head.params["selector.proj"] = torch.zeros(rank, head.cfg.hidden_size)
    head.params["selector.proj"][0, 0] = 1.0
    hidden = torch.zeros(1, 2, head.cfg.hidden_size)
    hidden[0, :, 0] = 1.0  # proj = e_0, so only codebook column 0 scores

    assert head.path(hidden, anchor=0, backend=get_backend()) == [12, 5]
    assert dc.top_k == 3


class _StubTrunk:
    """A readout that hands back fixed logits, so the walk is the only variable."""

    def __init__(self, logits):
        self.logits, self.cfg = logits, tiny()

    def _linear(self, backend, hidden, key):
        return self.logits.unsqueeze(0)


def test_unmapped_tensors_raise_instead_of_vanishing(tmp_path):
    """A DFlash2 checkpoint read through the NextN map must raise, not load a
    crippled head. The 11 conv and selector tensors mapped to None and were
    dropped without a word; the first draft then died on a KeyError far from the
    cause, and only because every conv weight is a direct subscript -- 19 of them,
    no .get() anywhere. Had one been tolerant, the head would have drafted garbage.
    """
    _tiny_head(tmp_path)  # writes model.safetensors + config.json
    head_file = tmp_path / "model.safetensors"

    # Negative control: the correct reader takes the same file with nothing dropped.
    from tilerl.dflash2 import _DFLASH2_TOP

    ok = read_head_params(head_file, _DFLASH2_TOP)
    assert any(k.endswith("attn_conv.proj") for k in ok), sorted(ok)[:5]
    assert "selector.pred" in ok and "selector.succ" in ok

    with pytest.raises(RuntimeError, match="map to no parameter") as e:
        read_head_params(head_file, _DRAFT_TOP)
    for name in ("candidate_selector", "attention_conv", "mlp_conv"):
        assert name in str(e.value), str(e.value)


def test_norm_fold_is_per_format(tmp_path):
    """The zero-centered +1 fold belongs to Qwen NextN alone.

    NextN norms are y = x*(1+w); DFlash2's are plain w*x. Every norm weight in
    Qwen3.8-27B-DFlash2 has mean +0.43..+2.65 where the trunk's sit at -0.03 —
    they are the multipliers already. Folding scales all of them silently, with
    none of the anti-correlated logits that made the reverse bug findable.
    """
    head = _tiny_head(tmp_path)
    assert torch.allclose(head.params["norm"], torch.full_like(head.params["norm"], _NORM))

    nextn = tmp_path / "nextn.safetensors"
    w = torch.full((8,), _NORM)
    save_file({"mtp.norm.weight": w, "mtp.pre_fc_norm_hidden.weight": w.clone()}, str(nextn))
    folded = read_head_params(nextn, _DRAFT_TOP)
    assert torch.allclose(folded["norm"], torch.full_like(w, _NORM + 1.0))


# --- the block drafter on the engine tick ------------------------------------
_PROMPT = [11, 42, 7, 99, 3, 56]


def _engine_run(head, spec, n):
    """Greedy ``n`` tokens through the engine, with and without the block drafter."""
    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore

    engine = build_engine(
        tiny(), head.trunk, get_backend(), num_blocks=64, num_slots=4, max_batch=4,
        max_total_tokens=256, draft=head if spec else None,
        decode_graph=False, prefix_store=NoPrefixStore(),
    )
    rid = engine.submit(_PROMPT, SamplingParams(temperature=0.0, max_new_tokens=n, seed=0))
    out: dict = {}
    for _ in range(4 * n):
        out.update(engine.poll())
        if len(out.get(rid, ())) >= n:
            break
        engine.step()
    return out[rid], engine.stats()


def test_block_drafter_on_the_engine_tick(tmp_path):
    """The engine's block-draft path must draft, and must not change the output.

    Two arms in one process on one trunk. The random head accepts almost
    nothing, so a second arm feeds the drafter the trunk's own continuation:
    that is the only arm where a whole block commits, which is what exercises
    the multi-token commit — ``select_step(n_ok)`` and the aux slice behind it.
    """
    head = _tiny_head(tmp_path)
    base, base_stats = _engine_run(head, False, 24)

    spec, stats = _engine_run(head, True, 24)
    assert spec == base, (spec, base)
    # Without this a drafter that silently never drafts passes the equality above.
    assert stats["spec_drafted"] > 0, stats

    # Oracle drafts: block_hidden carries the anchor's absolute position, so the
    # continuation the base arm produced can be read off by position.
    at: dict = {}
    real = head.block_hidden

    def block_hidden(ctx, ctx_pos, anchor, start, backend):
        at["start"] = start
        return real(ctx, ctx_pos, anchor, start, backend)

    def path(hidden, anchor, backend):
        i = at["start"] - len(_PROMPT) + 1
        # slot 0 is the token the trunk committed at ``start``; an off-by-one anchor
        # is invisible to output equality, so it is checked where it is handed over
        assert anchor == base[i - 1], (anchor, base[i - 1])
        return [base[i + j] if 0 <= i + j < len(base) else 0 for j in range(hidden.shape[1])]

    head.block_hidden, head.path = block_hidden, path
    oracle, ostats = _engine_run(head, True, 24)
    assert oracle == base, (oracle, base)
    assert ostats["spec_accepted"] > 0, ostats
    # The point of drafting: the same 24 tokens in fewer trunk forwards. An
    # anchor off by one still passes the equality checks and fails here.
    assert ostats["decode_forwards"] < base_stats["decode_forwards"], (ostats, base_stats)


def test_selector_codebooks_survive_the_engines_fp8_pass(tmp_path):
    """``_quantize_draft`` packs any [N,K] with both dims >= 128, and the two
    codebooks are [248320, 256] tables the walk indexes by token id. Shape
    cannot tell them from a projection, so the head names them."""
    from tilerl.engine import _quantize_draft

    head = _tiny_head(tmp_path)
    p = {k: torch.zeros(256, 256) for k in head.no_quant} | {"fc": torch.zeros(256, 256)}
    out = _quantize_draft(p, skip=head.no_quant)
    assert set(out) == {*head.no_quant, "fc.w8", "fc.wscale"}, sorted(out)


def test_load_draft_takes_a_checkpoint_directory(tmp_path):
    """A directory mmapped as a file raises OSError(ENODEV), which names nothing;
    three 27B runs were spent on it."""
    from tilerl.spec import load_draft

    _tiny_head(tmp_path)  # writes config.json + model.safetensors under tmp_path
    trunk = build_random(tiny(), seed=0)
    assert load_draft(trunk, tmp_path).width == load_draft(
        trunk, tmp_path / "model.safetensors").width
    (empty := tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no model.safetensors"):
        load_draft(trunk, empty)


def test_spec_depth_is_the_checkpoints_block(tmp_path):
    head = _tiny_head(tmp_path)
    block = _HF["dflash_config"]["block_size"]
    head.set_depth(None)  # None keeps the checkpoint's block
    assert head.width == block
    head.set_depth(block - 1)  # restating it is not a conflict
    assert head.width == block
    with pytest.raises(ValueError, match="verify width is fixed"):
        head.set_depth(2)


def test_engine_block_equals_the_full_context_block(tmp_path):
    """The block the engine drafts must equal the block ``draft()`` produces from
    the whole sequence. Output equality cannot see this — a rejected draft costs
    throughput, never tokens — so a starved or misaligned context is silent
    everywhere else.
    """
    import numpy as np

    from tilerl.engine import SamplingParams, build_engine
    from tilerl.kv_cache import NoPrefixStore
    from tilerl.train import _training_kv

    head, backend = _tiny_head(tmp_path), get_backend()
    model, taps = head.trunk, head.dcfg.target_layers
    engine = build_engine(
        tiny(), model, backend, num_blocks=64, num_slots=4, max_batch=4,
        max_total_tokens=256, max_num_batched_tokens=4, draft=head, decode_graph=False,
        prefix_store=NoPrefixStore(),  # 4: the prompt spans two prefill chunks
    )
    seen: list[tuple[list[int], list[int]]] = []
    inner = engine._draft.step

    def spy(rows):
        inner(rows)
        seen.extend((list(r.tokens), list(r.drafts)) for r in rows if r.drafts)

    engine._draft.step = spy
    rid = engine.submit(_PROMPT, SamplingParams(temperature=0.0, max_new_tokens=12, seed=0))
    for _ in range(48):
        if engine.poll().get(rid):
            break
        engine.step()

    assert len(seen) >= 3, seen
    for tokens, drafts in seen[:3]:
        ctx, pos = tokens[:-1], np.arange(len(tokens) - 1)
        hid: list = []
        model.forward(np.array([ctx]), pos, _training_kv(model, 1, len(ctx), device=backend.device),
                      backend, hidden_out=hid, aux_layers=taps)
        want = head.draft(torch.cat(hid[:-1], -1), pos, tokens[-1], backend)
        assert drafts == want, (len(ctx), drafts, want)
