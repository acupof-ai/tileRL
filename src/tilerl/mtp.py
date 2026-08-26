"""MTP draft head (model_mtp.safetensors): 1-layer DeepSeek-style next-token
head, batched over in-flight requests.

Draft forward per request: ``fc(concat(rmsnorm(emb(t)), rmsnorm(h)))`` ->
1 gated full-attn transformer layer -> norm -> shared lm_head -> draft
logits. The layer's attention is over the draft block only (level 0 = 1
row), so no per-sequence KV cache is needed — the head is stateless.

Weights are bf16 (the checkpoint ships 1 BF16 MTP layer, 849 MB); the
shared embed/lm_head come from the trunk (lm_head is fp4-packed, dispatched
through ``Model._linear``).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

#: HF tensor name -> param key (1 layer, index 0).
_KEY_MAP = {
    "mtp.fc.weight": "fc",
    "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden",
    "mtp.norm.weight": "norm",
    "mtp.layers.0.input_layernorm.weight": "input_layernorm",
    "mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm",
    "mtp.layers.0.self_attn.q_proj.weight": "q_proj",
    "mtp.layers.0.self_attn.k_proj.weight": "k_proj",
    "mtp.layers.0.self_attn.v_proj.weight": "v_proj",
    "mtp.layers.0.self_attn.o_proj.weight": "o_proj",
    "mtp.layers.0.self_attn.q_norm.weight": "q_norm",
    "mtp.layers.0.self_attn.k_norm.weight": "k_norm",
    "mtp.layers.0.mlp.gate_proj.weight": "gate_proj",
    "mtp.layers.0.mlp.up_proj.weight": "up_proj",
    "mtp.layers.0.mlp.down_proj.weight": "down_proj",
}


class MtpHead:
    """1-layer MTP draft head. ``params`` maps :data:`_KEY_MAP` values to
    bf16 tensors."""

    def __init__(self, params: dict[str, torch.Tensor]):
        self.params = params

    def forward(self, backend, hidden, token_ids, model):
        """Batched draft forward.

        ``hidden`` [B, hidden_size] trunk last-layer hidden that produced
        each pending token; ``token_ids`` [B] long pending tokens. Returns
        draft logits [B, vocab_size].
        """
        cfg = model.cfg
        p = self.params
        hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        B = hidden.shape[0]

        emb = backend.embedding(token_ids, model.params["embed_tokens"])
        emb_n = backend.rmsnorm(emb, p["pre_fc_norm_embedding"], cfg.rms_eps)
        h_n = backend.rmsnorm(hidden, p["pre_fc_norm_hidden"], cfg.rms_eps)
        x = backend.linear(torch.cat([emb_n, h_n], dim=-1), p["fc"])

        # Gated full-attn layer (same layout as the trunk's full-attn).
        h = backend.rmsnorm(x, p["input_layernorm"], cfg.rms_eps)
        q = backend.linear(h, p["q_proj"]).reshape(B, 1, hq, 2, d)
        gate = q[:, :, :, 1, :]
        q = q[:, :, :, 0, :]
        k = backend.linear(h, p["k_proj"]).reshape(B, 1, hkv, d)
        v = backend.linear(h, p["v_proj"]).reshape(B, 1, hkv, d)
        q = backend.rmsnorm(q, p["q_norm"], cfg.rms_eps)
        k = backend.rmsnorm(k, p["k_norm"], cfg.rms_eps)
        pos = torch.zeros(B, 1, dtype=torch.long, device=backend.device)
        q = backend.rope(q, pos, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        k = backend.rope(k, pos, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        out = backend.attention(q, k, v, 1.0 / math.sqrt(d), gate=gate)
        out = backend.linear(out.reshape(B, hq * d), p["o_proj"])
        x = backend.add(x, out)

        h = backend.rmsnorm(x, p["post_attention_layernorm"], cfg.rms_eps)
        gate_mlp = backend.linear(h, p["gate_proj"])
        up = backend.linear(h, p["up_proj"])
        down = backend.linear(backend.silu_mul(gate_mlp, up), p["down_proj"])
        x = backend.add(x, down)

        x = backend.rmsnorm(x, p["norm"], cfg.rms_eps)
        head_key = "embed_tokens" if cfg.tie_word_embeddings else "lm_head"
        return model._linear(backend, x, head_key)


def load_mtp_head(source: str) -> MtpHead:
    """Load ``model_mtp.safetensors`` from a checkpoint directory."""
    from safetensors.torch import load_file

    path = Path(source) / "model_mtp.safetensors"
    if not path.exists():
        raise RuntimeError(f"MTP head not found at {path}")
    tensors = load_file(str(path))
    params: dict[str, torch.Tensor] = {}
    for hf_name, tensor in tensors.items():
        key = _KEY_MAP.get(hf_name)
        if key is None:
            raise RuntimeError(f"unexpected MTP tensor `{hf_name}` (not in _KEY_MAP)")
        params[key] = tensor
    missing = set(_KEY_MAP.values()) - set(params)
    if missing:
        raise RuntimeError(f"MTP head missing tensors: {sorted(missing)}")
    return MtpHead(params)
