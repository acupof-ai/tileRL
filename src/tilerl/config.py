"""Model configs for tileRL.

``ModelConfig`` is the frozen description of a Qwen3.5/3.6-style hybrid model:
full-attention layers mixed with gated-delta linear-attention layers. The field
set mirrors agent-infer's ``qwen35-spec`` ``Qwen35Config``
(``crates/qwen35-spec/src/lib.rs``) minus the MoE/MTP/vision fields tileRL does
not implement.

Which layers are full-attention is a checkpoint property, never a hardcoded
index pattern: ``load_hf`` reads HF ``config.json`` ``layer_types`` verbatim and
validates it against ``full_attn_layers`` (see ``model.load_hf``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ModelConfig:
    # --- pinned contract fields -------------------------------------------
    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    full_attn_layers: tuple[int, ...]
    rope_theta: float
    max_position_embeddings: int
    rms_eps: float
    tie_word_embeddings: bool
    fp4: bool
    # --- extensions (defaults keep the pinned 14-field construction valid)
    #: Qwen3.5/3.6 full-attn q_proj carries a per-head sigmoid output gate:
    #: q_proj rows = num_heads * head_dim * 2, interleaved [query(HD); gate(HD)]
    #: per head. Vanilla Qwen3 has no gate.
    full_attn_gated: bool = True
    #: RoPE dimension (Qwen3.8: partial RoPE, 64 of head_dim 256). None -> head_dim.
    rotary_dim: int | None = None
    # --- gated-delta (linear attention) dimensions ------------------------
    linear_num_key_heads: int = 0
    linear_key_head_dim: int = 0
    linear_num_value_heads: int = 0
    linear_value_head_dim: int = 0
    linear_conv_kernel_dim: int = 4

    def __post_init__(self) -> None:
        # Accept lists, normalize to a sorted tuple.
        if not isinstance(self.full_attn_layers, tuple):
            object.__setattr__(self, "full_attn_layers", tuple(sorted(self.full_attn_layers)))
        full = self.full_attn_layers
        if not all(0 <= i < self.num_layers for i in full):
            raise ValueError(f"full_attn_layers {full} out of range for {self.num_layers} layers")
        if len(set(full)) != len(full):
            raise ValueError(f"full_attn_layers {full} has duplicates")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads}) for GQA"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {self.head_dim}")
        rd = self.effective_rotary_dim
        if rd > self.head_dim or rd % 2 != 0:
            raise ValueError(f"rotary_dim {rd} must be even and <= head_dim {self.head_dim}")
        if self.num_linear_layers > 0:
            if self.linear_num_value_heads % self.linear_num_key_heads != 0:
                raise ValueError(
                    "linear_num_value_heads must be divisible by linear_num_key_heads "
                    "(key_head = h * nkh / nvh must be an integer)"
                )
            for name, v in (
                ("linear_num_key_heads", self.linear_num_key_heads),
                ("linear_key_head_dim", self.linear_key_head_dim),
                ("linear_num_value_heads", self.linear_num_value_heads),
                ("linear_value_head_dim", self.linear_value_head_dim),
            ):
                if v <= 0:
                    raise ValueError(f"{name} must be positive when the model has linear layers")
            if self.linear_key_head_dim != self.linear_value_head_dim:
                # LinearStatePool shapes both recurrent state axes on the
                # value dim (kv_cache.py); a kd!=vd checkpoint would run with
                # a silently mis-shaped K axis.
                raise ValueError(
                    f"linear_key_head_dim ({self.linear_key_head_dim}) must equal "
                    f"linear_value_head_dim ({self.linear_value_head_dim})"
                )

    # --- derived quantities ----------------------------------------------
    @property
    def effective_rotary_dim(self) -> int:
        return self.rotary_dim if self.rotary_dim is not None else self.head_dim

    @property
    def full_attn_layer_set(self) -> FrozenSet[int]:
        return frozenset(self.full_attn_layers)

    @property
    def num_linear_layers(self) -> int:
        return self.num_layers - len(self.full_attn_layers)

    @property
    def linear_q_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_k_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_v_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def linear_qkv_dim(self) -> int:
        return self.linear_q_dim + self.linear_k_dim + self.linear_v_dim

    def is_full_attn(self, layer_idx: int) -> bool:
        return layer_idx in self.full_attn_layer_set


def qwen38_27b() -> ModelConfig:
    """Qwen3.8-27B (NVFP4): 64 layers = 16 full-attn (idx 3,7,...,63) + 48 GDN.

    Values verified against the /data00/Qwen3.8-27B-NVFP4 checkpoint: hidden
    5120, intermediate 17408, 24 attention heads / 4 KV heads / head_dim 256,
    vocab 248320, rope_theta 1e7, rms_eps 1e-6, untied (ships lm_head.weight),
    partial RoPE
    (rotary_dim 64), GDN 16 key heads / 48 value heads / key+value dim 128
    (A_log is [48]; same as the Qwen3.6 slices), conv kernel 4.
    The full-attn offset (i%4==3) follows the Qwen3.5 family convention
    (the checkpoint's layer_types agrees); load_hf validates the real
    checkpoint's layer_types against this tuple and raises on mismatch.
    """
    return ModelConfig(
        name="qwen38-27b",
        hidden_size=5120,
        intermediate_size=17408,
        num_layers=64,
        num_attention_heads=24,
        num_kv_heads=4,
        head_dim=256,
        vocab_size=248320,
        full_attn_layers=tuple(range(3, 64, 4)),
        rope_theta=1e7,
        max_position_embeddings=262144,
        rms_eps=1e-6,
        tie_word_embeddings=False,
        fp4=True,
        full_attn_gated=True,
        rotary_dim=64,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def qwen36_27b() -> ModelConfig:
    """Qwen3.6-27B (ModelOpt NVFP4, checkpoint /host/tc27-nvfp4-slice2): 64
    layers = 16 full-attn (idx 3,7,...,63) + 48 GDN. Same shapes and
    untied lm_head as :func:`qwen38_27b`; the checkpoint is ModelOpt NVFP4
    (its config.json says tie_word_embeddings=false and ships lm_head.weight).
    MLP linears are NVFP4 (weight_packed + f8 weight_scale + global scale,
    the stored global being its reciprocal), GDN in_proj_qkv/in_proj_z/
    out_proj are FP8 block-128 (weight + per-block scale_inv, multiplied);
    load_hf dequantizes both to bf16.
    """
    return ModelConfig(
        name="qwen36-27b",
        hidden_size=5120,
        intermediate_size=17408,
        num_layers=64,
        num_attention_heads=24,
        num_kv_heads=4,
        head_dim=256,
        vocab_size=248320,
        full_attn_layers=tuple(range(3, 64, 4)),
        rope_theta=1e7,
        max_position_embeddings=262144,
        rms_eps=1e-6,
        tie_word_embeddings=False,
        fp4=True,
        full_attn_gated=True,
        rotary_dim=64,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )


def tiny(max_position_embeddings: int = 512) -> ModelConfig:
    """CPU smoke-test config: 2 layers (idx 0 full-attn, idx 1 gated-delta).

    Small enough to forward on the CPU target in seconds; keeps every
    architectural feature of the 27B (gated q_proj, GDN conv1d, q/k-norm,
    partial-RoPE-capable) except fp4 (off by default).

    ``max_position_embeddings`` is a parameter because one real Claude Code
    turn does not fit in 512: measured 2026-09-02, a single Messages request
    carries ~5,310 tokens with 24 of its 28 tools disabled, nearly all of it
    harness preamble rather than task. `tiny-agent` is this config at 8192 --
    the cost is the KV pool, not parameters, since hidden is 64 and depth is 2.
    """
    return ModelConfig(
        name="tiny" if max_position_embeddings == 512 else "tiny-agent",
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=320,
        full_attn_layers=(0,),
        rope_theta=1e7,
        max_position_embeddings=max_position_embeddings,
        rms_eps=1e-6,
        tie_word_embeddings=True,
        fp4=False,
        full_attn_gated=True,
        rotary_dim=16,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=2,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
