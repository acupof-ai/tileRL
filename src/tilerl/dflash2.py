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
from .model import Model
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

    #: codebooks the selector GATHERS rows from, not linears: block-quantizing
    #: them would replace [248320,256] with a .w8 the walk cannot index.
    no_quant = ("selector.pred", "selector.succ")

    def __init__(self, trunk: Any, params: dict[str, torch.Tensor], cfg, dcfg) -> None:
        self.trunk, self.params, self.cfg, self.dcfg = trunk, params, cfg, dcfg
        self.groups = cfg.hidden_size // dcfg.group_size
        #: the engine re-serves the head fp8; a Model gives the head the same
        #: .w8/.wscale dispatch the trunk has, over THIS params dict.
        self.layers = Model(cfg, params)
        #: the block is fixed by the checkpoint: anchor + block_size-1 drafts
        self.width = dcfg.block_size  # verify tick: anchor + block_size-1 drafts
        self.aux_layers = dcfg.target_layers

    def set_depth(self, depth: int | None) -> None:
        """The block is the checkpoint's, so ``spec_depth`` is not the caller's to choose."""
        if depth is not None and depth != self.width - 1:
            raise ValueError(
                f"spec_depth={depth} with a block drafter: the checkpoint's block is "
                f"{self.width} slots (anchor + {self.width - 1} drafts), so the verify "
                f"width is fixed. Pass spec_depth=None."
            )

    def attach(self, backend, num_blocks: int, dtype=None) -> None:
        """No KV plane of its own: the context K/V is projected from the trunk's aux taps,
        so ``dtype`` (the trunk pool's, part of the contract) has nothing to size here."""
        self.backend = backend

    def step(self, rows) -> None:
        """Leave next tick's chain in ``r.drafts``. A context position's K/V is a
        pure projection of the trunk's aux taps, so it is projected the tick it
        commits and kept — re-projecting the whole context every tick is ~10 ms at
        T=512 B=8. The walk runs once for the tick; ``block_hidden`` is still per
        row, which is the rest of the batching ceiling."""
        backend = self.backend
        dev = backend.device
        live, hidden, anchors = [], [], []
        for r in rows:
            if r.aux is None or r.done:
                continue
            # The context is every position but the anchor; a row still prefilling has none.
            end = r.seq_len if r.prefilling else r.seq_len - 1
            lo, base = r.ctx_len, r.hidden_from
            if lo < base:
                raise RuntimeError(
                    f"req {r.req_id}: context K/V has a hole at {lo}..{base} — the trunk "
                    f"never forwarded those positions this process"
                )
            if end > lo:
                pos = torch.arange(lo, end, device=dev)
                new = self.context_kv(r.aux[:, lo - base : end - base], pos, backend)
                r.ctx = new if r.ctx is None else [
                    (torch.cat([k, nk], 1), torch.cat([v, nv], 1))
                    for (k, v), (nk, nv) in zip(r.ctx, new)
                ]
                w = self.dcfg.sliding_window
                if r.ctx[0][0].shape[1] > w:
                    r.ctx = [(k[:, -w:], v[:, -w:]) for k, v in r.ctx]
                r.ctx_len = end
            if not r.decoding or r.ctx is None:
                continue
            n = r.ctx[0][0].shape[1]
            ctx_pos = torch.arange(r.ctx_len - n, r.ctx_len, device=dev)
            anchor = r.tokens[-1]
            live.append(r)
            anchors.append(anchor)
            hidden.append(self.block_hidden(r.ctx, ctx_pos, anchor, r.ctx_len, backend))
        if live:
            walked = self.paths(torch.cat(hidden)[:, 1:], anchors, backend)
            for r, drafts in zip(live, walked):
                r.drafts = drafts


    def context_kv(self, aux_hidden, positions, backend) -> list[tuple]:
        """Per-layer (k, v) for the context, straight off the trunk's stacked taps."""
        cfg = self.cfg
        h = backend.rmsnorm(
            self._lin(backend, aux_hidden, "fc"), self.params["hidden_norm"], cfg.rms_eps
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
                self._lin(backend, h, f"{p}.gate_proj"),
                self._lin(backend, h, f"{p}.up_proj"),
            )
            h = self._lin(backend, h, f"{p}.down_proj")
            x = backend.add(x, self._conv(h, c, self.params[f"{p}.mlp_conv.base"][1]))
        return backend.rmsnorm(x, self.params["norm"], cfg.rms_eps)

    def path(self, hidden, anchor, backend) -> list[int]:
        return self.paths(hidden, [anchor], backend)[0]

    def paths(self, hidden, anchors, backend) -> list[list[int]]:
        """Top-k per slot, then one greedy walk per row. ``prev`` feeds the next
        slot so the block is sequential; the rows are not, so they walk together
        and the batch costs one host sync, not two per slot per row."""
        dc = self.dcfg
        head = self.trunk.cfg.head_key
        # topk before the widening cast: it is order-preserving, and an f32 copy of
        # the full [B, W, 248320] readout is 55 MB at B=8
        unary, cand = torch.topk(self.trunk._linear(backend, hidden, head), dc.top_k, dim=-1)
        proj = self._lin(backend, hidden, "selector.proj").float()
        # gathered rows only: an f32 copy of either [248320,256] codebook is 254 MB
        pred, succ = self.params["selector.pred"], self.params["selector.succ"]
        prev = torch.as_tensor(anchors, dtype=torch.long, device=hidden.device)
        rows = torch.arange(len(anchors), device=hidden.device)
        out = []
        for j in range(hidden.shape[1]):
            score = unary[:, j].float() + torch.einsum(
                "bkr,br->bk", succ[cand[:, j]].float(), pred[prev].float() * proj[:, j]
            )
            prev = cand[rows, j, score.argmax(-1)]
            out.append(prev)
        return torch.stack(out, 1).tolist()

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
    def _lin(self, backend, x, key):
        return self.layers._linear(backend, x, key)

    def _heads(self, backend, h, key, heads):
        return self._lin(backend, h, key).reshape(*h.shape[:2], heads, self.cfg.head_dim)

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
        return self._lin(backend, out.reshape(*h.shape[:2], -1), f"{p}.o_proj")

    def _conv_in(self, backend, x, norm_key, conv_key):
        """Norm, then the conv before the op — and the coefficients its partner
        after the op consumes. One projection yields both sides' taps."""
        h = backend.rmsnorm(x, self.params[norm_key], self.cfg.rms_eps)
        c = self._lin(backend, h, f"{conv_key}.proj").reshape(
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
            out[:, tap:] += coef[:, tap:, tap] * blocks[:, :-tap]
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
