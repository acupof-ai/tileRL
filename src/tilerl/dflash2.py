"""DFlash2 block drafter: one pass proposes a whole block, not a chain.

The draft never runs a layer over the prompt. A context position's K/V is a pure
projection of the trunk's hidden state there, so the block attends over the whole
context for the price of that projection. The block is the verified anchor token
plus ``block_size - 1`` mask slots attending bidirectionally, and a two-tap
grouped dynamic depthwise convolution around each attention and MLP is what lets
a later slot see the earlier ones without a second pass. The trunk's lm_head then
offers ``top_k`` candidates per slot, two rank-256 codebooks score adjacent
transitions, and the walk takes one path from the anchor.

z-lab/Qwen3.8-27B-DFlash2; the math mirrors vLLM's ``qwen3_dflash2.py`` (PR 52816).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .spec import read_head_params


@dataclass(frozen=True)
class DFlash2Config:
    """``config.json``'s ``dflash_config``, plus the window the draft attends over."""

    block_size: int
    taps: int
    group_size: int
    mask_token_id: int
    rank: int
    top_k: int
    #: trunk layers whose hidden states ``fc`` consumes, in concat order
    target_layers: tuple[int, ...]
    sliding_window: int


#: DFlash2 tensor stems -> param keys (see :func:`tilerl.spec.read_head_params`).
_DFLASH2_TOP = {
    "fc": "fc",
    "norm": "norm",
    "hidden_norm": "hidden_norm",
    "candidate_selector.hidden_projection": "selector.proj",
    "candidate_selector.predecessor_codebook": "selector.pred",
    "candidate_selector.successor_codebook": "selector.succ",
    "attention_conv.base_kernel": "attn_conv.base",
    "attention_conv.kernel_projection": "attn_conv.proj",
    "mlp_conv.base_kernel": "mlp_conv.base",
    "mlp_conv.kernel_projection": "mlp_conv.proj",
}

#: Knobs this port does not implement; each would change the draft with no error.
#: Qwen3.8-27B-DFlash2 sets none, so the raise is the whole implementation.
_UNSUPPORTED = (
    "input_embedding_scale",
    "output_multiplier",
    "final_logit_softcapping",
    "attention_sink_bias",
    "add_swa_attention_sink_bias",
)


def _attend(q, k, v, q_pos, k_pos, window):
    """Non-causal GQA: every block slot sees every other and every context key
    inside the window. ponytail: torch-eager, tilelang once the block draft is
    on the hot path."""
    d, group = q.shape[-1], q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(group, 2).float(), v.repeat_interleave(group, 2).float()
    att = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(d)
    att = att.masked_fill(q_pos[:, None] - k_pos[None, :] >= window, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(att, -1), v)


class DFlash2Head:
    """``params`` holds the head's own weights only: the embedding and the
    readout are the trunk's, and the block's K/V never outlives one draft."""

    def __init__(self, trunk: Any, params: dict[str, torch.Tensor], cfg, dcfg) -> None:
        self.trunk, self.params, self.cfg, self.dcfg = trunk, params, cfg, dcfg
        self.groups = cfg.hidden_size // dcfg.group_size

    def context_kv(self, aux_hidden, positions, backend) -> list[tuple]:
        """Per-layer (k, v) for the context, straight off the trunk's stacked taps."""
        cfg = self.cfg
        h = backend.rmsnorm(
            backend.linear(aux_hidden, self.params["fc"]), self.params["hidden_norm"], cfg.rms_eps
        )
        out = []
        for i in range(cfg.num_layers):
            k = self._heads(backend, h, f"layers.{i}.k_proj", cfg.num_kv_heads)
            k = backend.rmsnorm(k, self.params[f"layers.{i}.k_norm"], cfg.rms_eps)
            v = self._heads(backend, h, f"layers.{i}.v_proj", cfg.num_kv_heads)
            out.append((self._rope(backend, k, positions), v))
        return out

    def block_hidden(self, ctx, ctx_pos, anchor, start, backend) -> torch.Tensor:
        """The block's post-norm hidden ``[1, block_size, H]``: slot 0 is the
        anchor the trunk already committed, the rest are mask slots."""
        cfg, dc = self.cfg, self.dcfg
        dev = backend.device
        ids = torch.full((1, dc.block_size), dc.mask_token_id, dtype=torch.long, device=dev)
        ids[0, 0] = anchor
        pos = torch.arange(start, start + dc.block_size, device=dev)
        x = backend.embedding(ids, self.trunk.params["embed_tokens"])
        for i in range(cfg.num_layers):
            p = f"layers.{i}"
            h, c = self._conv_in(backend, x, f"{p}.input_norm", f"{p}.attn_conv")
            h = self._attn(backend, i, h, pos, ctx[i], ctx_pos)
            x = backend.add(x, self._conv(h, c, self.params[f"{p}.attn_conv.base"][1]))
            h, c = self._conv_in(backend, x, f"{p}.post_attn_norm", f"{p}.mlp_conv")
            h = backend.silu_mul(
                backend.linear(h, self.params[f"{p}.gate_proj"]),
                backend.linear(h, self.params[f"{p}.up_proj"]),
            )
            h = backend.linear(h, self.params[f"{p}.down_proj"])
            x = backend.add(x, self._conv(h, c, self.params[f"{p}.mlp_conv.base"][1]))
        return backend.rmsnorm(x, self.params["norm"], cfg.rms_eps)

    def path(self, hidden, anchor, backend) -> list[int]:
        """Top-k candidates per mask slot, then the best-scoring path from the
        anchor. The walk only ever reads the row of the transition score whose
        predecessor is the token it just emitted, so the row is all we build."""
        dc = self.dcfg
        head = "embed_tokens" if self.trunk.cfg.tie_word_embeddings else "lm_head"
        unary, cand = torch.topk(
            self.trunk._linear(backend, hidden, head)[0].float(), dc.top_k, dim=-1
        )
        proj = backend.linear(hidden, self.params["selector.proj"])[0].float()
        # gathered rows only: an f32 copy of either [248320,256] codebook is 254 MB
        pred, succ = self.params["selector.pred"], self.params["selector.succ"]
        out, prev = [], anchor
        for j in range(hidden.shape[1]):
            score = unary[j] + succ[cand[j]].float() @ (pred[prev].float() * proj[j])
            prev = int(cand[j, int(score.argmax())])
            out.append(prev)
        return out

    def draft(self, aux_hidden, positions, anchor, backend) -> list[int]:
        """``aux_hidden`` [1,T,len(target_layers)*H] concatenated in
        ``target_layers`` order and ``positions`` [T] describe the context;
        returns ``block_size - 1`` tokens continuing from ``anchor``, the token
        the trunk committed at ``positions[-1] + 1``."""
        pos = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        ctx = self.context_kv(aux_hidden, pos, backend)
        h = self.block_hidden(ctx, pos, anchor, int(pos[-1]) + 1, backend)
        return self.path(h[:, 1:], anchor, backend)

    # --- pieces -------------------------------------------------------------
    def _heads(self, backend, h, key, heads):
        return backend.linear(h, self.params[key]).reshape(*h.shape[:2], heads, self.cfg.head_dim)

    def _rope(self, backend, x, positions):
        cfg = self.cfg
        return backend.rope(x, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)

    def _attn(self, backend, i, h, pos, ctx, ctx_pos):
        cfg, p = self.cfg, f"layers.{i}"
        q = self._heads(backend, h, f"{p}.q_proj", cfg.num_attention_heads)
        k = self._heads(backend, h, f"{p}.k_proj", cfg.num_kv_heads)
        v = self._heads(backend, h, f"{p}.v_proj", cfg.num_kv_heads)
        q = self._rope(backend, backend.rmsnorm(q, self.params[f"{p}.q_norm"], cfg.rms_eps), pos)
        k = self._rope(backend, backend.rmsnorm(k, self.params[f"{p}.k_norm"], cfg.rms_eps), pos)
        k, v = torch.cat([ctx[0], k], 1), torch.cat([ctx[1], v], 1)
        out = _attend(q, k, v, pos, torch.cat([ctx_pos, pos]), self.dcfg.sliding_window)
        return backend.linear(out.reshape(*h.shape[:2], -1), self.params[f"{p}.o_proj"])

    def _conv_in(self, backend, x, norm_key, conv_key):
        """Norm, then the conv before the op — and the coefficients its partner
        after the op consumes. One projection yields both sides' taps."""
        h = backend.rmsnorm(x, self.params[norm_key], self.cfg.rms_eps)
        c = backend.linear(h, self.params[f"{conv_key}.proj"]).reshape(
            *h.shape[:2], 2, self.dcfg.taps, self.groups
        )
        return self._conv(h, c[:, :, 0], self.params[f"{conv_key}.base"][0]), c[:, :, 1]

    def _conv(self, x, delta, base):
        """Two-tap grouped depthwise conv along the block. Coefficients are
        per-token; a tap reaching before the block start is zero-padded, so slot
        0 carries only its own term and no earlier block leaks in."""
        dc, g = self.dcfg, self.groups
        blocks = x.reshape(*x.shape[:2], g, dc.group_size).float()
        coef = base.float().reshape(1, 1, dc.taps, g, dc.group_size) + delta.float().unsqueeze(-1)
        out = coef[:, :, 0] * blocks
        for tap in range(1, dc.taps):
            pad = torch.zeros_like(blocks[:, :tap])
            out = out + coef[:, :, tap] * torch.cat([pad, blocks[:, :-tap]], 1)
        return out.reshape(*x.shape[:2], -1)


def load_dflash2(trunk: Any, path: str | Path) -> DFlash2Head:
    """Load the block drafter from its safetensors and the config.json beside
    it. Its shapes are not the trunk's: 32/8 heads of 128, no attention gate,
    full RoPE."""
    path = Path(path)
    hf = json.loads((path.parent / "config.json").read_text())
    d = hf["dflash_config"]
    knobs = {**hf, **d}
    unsupported = [k for k in _UNSUPPORTED if knobs.get(k)]
    if unsupported:
        raise RuntimeError(f"DFlash2 head {path}: unimplemented {sorted(unsupported)}")
    if hf.get("is_causal") is not False:
        raise RuntimeError(f"DFlash2 head {path}: only non-causal block attention is implemented")
    n = hf["num_hidden_layers"]
    cfg = ModelConfig(
        name="dflash2",
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["intermediate_size"],
        num_layers=n,
        num_attention_heads=hf["num_attention_heads"],
        num_kv_heads=hf["num_key_value_heads"],
        head_dim=hf["head_dim"],
        vocab_size=hf["vocab_size"],
        full_attn_layers=tuple(range(n)),
        rope_theta=hf["rope_parameters"]["rope_theta"],
        max_position_embeddings=hf["max_position_embeddings"],
        rms_eps=hf["rms_norm_eps"],
        tie_word_embeddings=hf["tie_word_embeddings"],
        fp4=False,
        full_attn_gated=False,
    )
    dcfg = DFlash2Config(
        block_size=d["block_size"],
        taps=d["conv_kernel_size"],
        group_size=d["conv_group_size"],
        mask_token_id=d["mask_token_id"],
        rank=d["selector_rank"],
        top_k=d["selector_top_k"],
        target_layers=tuple(d["target_layer_ids"]),
        sliding_window=hf["sliding_window"],
    )
    return DFlash2Head(trunk, read_head_params(path, _DFLASH2_TOP), cfg, dcfg)
