"""Qwen3.5/3.6 hybrid model (full-attention + gated-delta layers) on backend
ops only, plus HF/MLX/NVFP4/FP8/AWQ checkpoint loading and saving. Layer order
and tensor names follow agent-infer's ``qwen35_forward.rs`` / ``qwen35-spec``."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tilerl_kernels.reference import (
    dequant_awq,
    dequant_fp8,
    dequant_nvfp4,
    pack_fp4,
    renorm_fp4_scale,
    unpack_fp4,
    untwiddle_fp4,
)

from . import autograd, precision
from .config import ModelConfig

if TYPE_CHECKING:  # pragma: no cover - typing only, no tilelang import at runtime
    import numpy as np
    from tilerl_kernels.backend import Backend


# --- Param schema -----------------------------------------------------------
def param_specs(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Canonical param keys + shapes: ``build_random`` draws them, ``load_hf`` validates."""
    h, inter, v = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
    specs: dict[str, tuple[int, ...]] = {
        "embed_tokens": (v, h),
        "final_norm": (h,),
    }
    if not cfg.tie_word_embeddings:
        specs["lm_head"] = (v, h)
    for i in range(cfg.num_layers):
        p = f"layers.{i}"
        specs[f"{p}.input_norm"] = (h,)
        specs[f"{p}.post_attn_norm"] = (h,)
        if cfg.is_full_attn(i):
            q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
            specs[f"{p}.q_proj"] = (q_rows, h)
            specs[f"{p}.k_proj"] = (hkv * d, h)
            specs[f"{p}.v_proj"] = (hkv * d, h)
            specs[f"{p}.o_proj"] = (h, hq * d)
            specs[f"{p}.q_norm"] = (d,)
            specs[f"{p}.k_norm"] = (d,)
        else:
            qd, kd, vd = cfg.linear_q_dim, cfg.linear_k_dim, cfg.linear_v_dim
            nvh = cfg.linear_num_value_heads
            specs[f"{p}.in_proj_qkv"] = (qd + kd + vd, h)
            specs[f"{p}.in_proj_z"] = (vd, h)
            specs[f"{p}.in_proj_b"] = (nvh, h)
            specs[f"{p}.in_proj_a"] = (nvh, h)
            specs[f"{p}.conv1d"] = (qd + kd + vd, cfg.linear_conv_kernel_dim)
            specs[f"{p}.dt_bias"] = (nvh,)
            specs[f"{p}.a_log"] = (nvh,)
            specs[f"{p}.gdn_norm"] = (cfg.linear_value_head_dim,)
            specs[f"{p}.out_proj"] = (h, vd)
        specs[f"{p}.gate_proj"] = (inter, h)
        specs[f"{p}.up_proj"] = (inter, h)
        specs[f"{p}.down_proj"] = (h, inter)
    return specs


def fp4_param_keys(cfg: ModelConfig) -> set[str]:
    """The projection keys that are fp4-packed when cfg.fp4 (conv1d's K=4 < block 16 stays bf16)."""
    keys: set[str] = set()
    for i in range(cfg.num_layers):
        p = f"layers.{i}"
        if cfg.is_full_attn(i):
            keys |= {f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj", f"{p}.o_proj"}
        else:
            keys |= {
                f"{p}.in_proj_qkv",
                f"{p}.in_proj_z",
                f"{p}.in_proj_b",
                f"{p}.in_proj_a",
                f"{p}.out_proj",
            }
        keys |= {f"{p}.gate_proj", f"{p}.up_proj", f"{p}.down_proj"}
    if not cfg.tie_word_embeddings:
        keys.add("lm_head")
    return keys


def _quantized(params: dict[str, torch.Tensor], key: str) -> bool:
    return f"{key}.wq" in params or f"{key}.w8" in params


def _projection_groups(cfg: ModelConfig, layer_idx: int) -> list[tuple[str, list[str]]]:
    """Same-input projection groups: one GEMM per group replaces the launches
    of the small projections the decode tick is latency-bound on. Serving only."""
    p = f"layers.{layer_idx}"
    groups = [(f"{p}.gate_up", [f"{p}.gate_proj", f"{p}.up_proj"])]
    if cfg.is_full_attn(layer_idx):
        groups.append((f"{p}.qkv", [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"]))
    else:
        groups.append((f"{p}.ab", [f"{p}.in_proj_a", f"{p}.in_proj_b"]))
        groups.append((f"{p}.qkvz", [f"{p}.in_proj_qkv", f"{p}.in_proj_z"]))
    return groups


def _fuse_projections(cfg: ModelConfig, params: dict[str, torch.Tensor]) -> None:
    """Concat each group's packed weights (fp4 .wq/.scale/.oscale, fp8
    .w8/.wscale/.oscale) into a fused key in place; any bf16 master stays."""
    for i in range(cfg.num_layers):
        for fused_key, group in _projection_groups(cfg, i):
            if _quantized(params, fused_key):
                continue
            if all(f"{k}.wq" in params for k in group):
                for suffix in (".wq", ".scale", ".oscale"):
                    params[fused_key + suffix] = torch.cat(
                        [params.pop(k + suffix) for k in group]).contiguous()
                continue
            if not all(f"{k}.w8" in params for k in group):
                continue
            if any(params[f"{k}.w8"].shape[0] % 128 for k in group[:-1]):
                continue  # the per-128-block wscale grid would not concat losslessly
            for suffix in (".w8", ".wscale"):
                params[fused_key + suffix] = torch.cat(
                    [params.pop(k + suffix) for k in group]).contiguous()
            if f"{group[0]}.oscale" in params:
                params[f"{fused_key}.oscale"] = torch.cat(
                    [params[f"{k}.oscale"] for k in group]).contiguous()
            for k in group:
                params.pop(f"{k}.oscale", None)


def _native_fp4(packed, weight_scale, gscale, *, divide: bool = False):
    """NVFP4 checkpoint tensors -> (wq, scale, oscale), nibbles served verbatim."""
    gs = gscale.float().reshape(1)
    gs = 1.0 / gs if divide else gs  # ModelOpt stores the global scale's reciprocal
    scale, oscale = renorm_fp4_scale(weight_scale.float(), gs.expand(packed.shape[0]))
    return packed.contiguous(), scale, oscale


# --- Model ------------------------------------------------------------------
class Model:
    """``params`` maps :func:`param_specs` keys to bf16 tensors; quantized
    linears carry ``<key>.wq/.scale`` (fp4) or ``<key>.w8/.wscale`` (fp8) plus an
    optional per-row ``<key>.oscale``, with a bf16 master beside them only for training."""

    def __init__(self, cfg: ModelConfig, params: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.params = params

    def _has(self, key: str) -> bool:
        return key in self.params or _quantized(self.params, key)

    def _linear(self, backend: Backend, x: torch.Tensor, key: str, residual=None) -> torch.Tensor:
        y = self._base_linear(backend, x, key, residual)
        a = self.params.get(key + ".lora_a")
        if a is None:
            return y
        return backend.add(y, backend.linear(backend.linear(x, a), self.params[key + ".lora_b"]))

    def _base_linear(self, backend: Backend, x: torch.Tensor, key: str, residual=None):
        # ``master`` is recording-only: the STE grad lands on it.
        kw = {} if residual is None else {"residual": residual}
        wq, osc = self.params.get(key + ".wq"), self.params.get(key + ".oscale")
        if wq is not None:
            scale = self.params[key + ".scale"]
            return backend.linear_fp4(x, wq, scale, master=self.params.get(key), oscale=osc, **kw)
        w8 = self.params.get(key + ".w8")
        if w8 is not None:
            wscale = self.params[key + ".wscale"]
            return backend.linear_fp8(x, w8, wscale, master=self.params.get(key), oscale=osc, **kw)
        return backend.linear(x, self.params[key], **kw)

    def _add_via(self, backend: Backend, kv: Any, x: torch.Tensor, h: torch.Tensor, key: str):
        """x + linear(h): in the GEMV epilogue when serving, backend.add otherwise."""
        if getattr(backend, "tp_world", 1) > 1:
            # Row-parallel: the residual add has to follow the all-reduce.
            return backend.add(x, backend.all_reduce(self._linear(backend, h, key)))
        if getattr(backend, "fuses_residual", False) and not getattr(kv, "dense", False):
            return self._linear(backend, h, key, residual=x)
        return backend.add(x, self._linear(backend, h, key))

    def _full_attn(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv: Any,
        backend: Backend,
    ) -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.input_norm"], cfg.rms_eps)
        hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        qkv_key = f"{p}.qkv"
        if self._has(qkv_key):
            qkv = self._linear(backend, h, qkv_key)
            q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
            b, t, _ = qkv.shape
            qn = None
            if cfg.full_attn_gated and not getattr(kv, "dense", False):
                qn = backend.attn_prep(
                    qkv, self.params[f"{p}.q_norm"], self.params[f"{p}.k_norm"], positions,
                    cfg.rope_theta, cfg.effective_rotary_dim, kv, layer_idx, hq, hkv, cfg.rms_eps,
                )
            if qn is not None:  # sm90: norm+rope+kv-write in one launch
                gate = autograd.slice(autograd.reshape(
                    autograd.slice(qkv, ..., slice(0, q_rows)), b, t, hq, 2, d), ..., 1, slice(None))
                k_plane, v_plane = kv.kv_pool.kv_layer(layer_idx)
                out = backend.paged_attention(
                    qn, k_plane, v_plane, kv.block_table, kv.seq_len, 1.0 / math.sqrt(d),
                    gate=gate, seq_q_lens=getattr(kv, "seq_q_lens", None),
                )
                return self._add_via(backend, kv, x, autograd.reshape(out, b, t, hq * d), f"{p}.o_proj")
            q = autograd.slice(qkv, ..., slice(0, q_rows))
            k = autograd.slice(qkv, ..., slice(q_rows, q_rows + hkv * d))
            v = autograd.slice(qkv, ..., slice(q_rows + hkv * d, None))
        else:
            q = self._linear(backend, h, f"{p}.q_proj")
            k = self._linear(backend, h, f"{p}.k_proj")
            v = self._linear(backend, h, f"{p}.v_proj")
        b, t, _ = q.shape
        if cfg.full_attn_gated:
            # q_proj rows interleave [query(HD); gate(HD)] per head.
            q = autograd.reshape(q, b, t, hq, 2, d)
            gate = autograd.slice(q, ..., 1, slice(None))  # [b,t,hq,d]
            q = autograd.slice(q, ..., 0, slice(None))  # [b,t,hq,d]
        else:
            gate = None
            q = autograd.reshape(q, b, t, hq, d)
        k = autograd.reshape(k, b, t, hkv, d)
        v = autograd.reshape(v, b, t, hkv, d)
        # f32 out: these feed rope and then the bf16 KV pool, so a bf16 store here
        # would round twice. input_norm/post_attn_norm/final_norm keep bf16 -- their
        # consumers requantize (errors/2026-09-03-unfused-prelude-double-rounds.md).
        q = backend.rmsnorm_f32(q, self.params[f"{p}.q_norm"], cfg.rms_eps)
        k = backend.rmsnorm_f32(k, self.params[f"{p}.k_norm"], cfg.rms_eps)
        q = backend.rope(q, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        k = backend.rope(k, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        if getattr(kv, "dense", False):
            out = backend.attention(q, k, v, 1.0 / math.sqrt(d), gate=gate)
        else:
            backend.write_tokens(k, v, kv, layer_idx)
            k_plane, v_plane = kv.kv_pool.kv_layer(layer_idx)
            out = backend.paged_attention(
                q,
                k_plane,
                v_plane,
                kv.block_table,
                kv.seq_len,
                1.0 / math.sqrt(d),
                gate=gate,
                seq_q_lens=getattr(kv, "seq_q_lens", None),
            )
        return self._add_via(backend, kv, x, autograd.reshape(out, b, t, hq * d), f"{p}.o_proj")

    def _gdn(
        self,
        layer_idx: int,
        linear_idx: int,
        x: torch.Tensor,
        kv: Any,
        backend: Backend,
    ) -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.input_norm"], cfg.rms_eps)
        qkvz_key = f"{p}.qkvz"
        if self._has(qkvz_key):
            qkvz = self._linear(backend, h, qkvz_key)
            qkv = autograd.slice(qkvz, ..., slice(0, cfg.linear_qkv_dim))
            z = autograd.slice(qkvz, ..., slice(cfg.linear_qkv_dim, None))
        else:
            qkv = self._linear(backend, h, f"{p}.in_proj_qkv")
            z = self._linear(backend, h, f"{p}.in_proj_z")
        ab_key = f"{p}.ab"
        if self._has(ab_key):
            ab = self._linear(backend, h, ab_key)
            nvh = cfg.linear_num_value_heads
            a_proj = autograd.slice(ab, ..., slice(0, nvh))
            b_proj = autograd.slice(ab, ..., slice(nvh, None))
        else:
            b_proj = self._linear(backend, h, f"{p}.in_proj_b")
            a_proj = self._linear(backend, h, f"{p}.in_proj_a")
        qd, kd = cfg.linear_q_dim, cfg.linear_k_dim
        q = autograd.slice(qkv, ..., slice(0, qd))
        k = autograd.slice(qkv, ..., slice(qd, qd + kd))
        v = autograd.slice(qkv, ..., slice(qd + kd, None))
        kwargs = dict(
            z=z,
            conv1d_weight=self.params[f"{p}.conv1d"],
            dt_bias=self.params[f"{p}.dt_bias"],
            a_log=self.params[f"{p}.a_log"],
            norm_weight=self.params[f"{p}.gdn_norm"],
            seq_q_lens=getattr(kv, "seq_q_lens", None),
        )
        # Speculative verify keeps the state after every chain step for the engine to adopt.
        ks = getattr(kv, "keep_steps", 0)
        out = None
        if not getattr(kv, "dense", False):
            out = backend.gdn_decode(  # sm90: in-place pool state, one launch
                q, k, v, a_proj, b_proj, kv.state_pool, kv.state_slot, linear_idx,
                keep_steps=ks, **kwargs
            )
        if out is None:
            pool = kv.state_pool
            state, window = backend.state_gather(
                pool.states, pool.conv_windows, kv.state_slot, linear_idx, pool.win_parity
            )
            if ks:  # the tape replays kwargs into gdn_backward, which has no such arg
                kwargs["keep_steps"] = ks
            out, new_state, new_window = backend.linear_attn_chunk(
                q, k, v, a_proj, b_proj, state, conv_window=window, **kwargs
            )
            backend.state_scatter(
                pool.step_states if ks else pool.states,
                pool.step_windows if ks else pool.conv_windows,
                kv.state_slot, linear_idx, new_state, new_window,
                None if ks else pool.win_parity, steps=bool(ks),
            )
        elif linear_idx == cfg.num_linear_layers - 1:
            # every GDN layer of this tick read plane p and wrote 1-p: flip once
            backend.flip_window_parity(kv.state_pool, kv.state_slot)
        return self._add_via(backend, kv, x, out, f"{p}.out_proj")

    def _mlp(self, layer_idx: int, x: torch.Tensor, kv: Any, backend: Backend) -> torch.Tensor:
        # The layer's largest pure block: attention and GDN advance the pools, so
        # replaying either would recompute against state its own forward moved.
        return autograd.checkpoint(self._mlp_body, layer_idx, x, kv, backend)

    def _mlp_body(self, layer_idx: int, x: torch.Tensor, kv: Any, backend: Backend):
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.post_attn_norm"], cfg.rms_eps)
        gu_key = f"{p}.gate_up"
        if self._has(gu_key):
            gu = self._linear(backend, h, gu_key)
            gate = autograd.slice(gu, ..., slice(0, cfg.intermediate_size))
            up = autograd.slice(gu, ..., slice(cfg.intermediate_size, None))
        else:
            gate = self._linear(backend, h, f"{p}.gate_proj")
            up = self._linear(backend, h, f"{p}.up_proj")
        activated = backend.silu_mul(gate, up)
        return self._add_via(backend, kv, x, activated, f"{p}.down_proj")

    def forward(
        self,
        input_ids: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
        kv: Any,
        backend: Backend,
        hidden_out: list | None = None,
        last_only: bool | list[int] = False,
        aux_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """``input_ids`` [B,T], ``positions`` [T] or [B,T], ``kv`` a BatchKv with
        pools attached -> logits [B,T,vocab]. ``hidden_out`` receives each
        ``aux_layers`` layer's output in order (the DFlash2 drafter's fc input),
        then the pre-final-norm hidden state (the MTP draft head's input)."""
        cfg = self.cfg
        device = backend.device
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        pos = torch.as_tensor(positions, dtype=torch.long, device=device)
        x = backend.embedding(ids, self.params["embed_tokens"])
        linear_idx = 0
        for i in range(cfg.num_layers):
            if cfg.is_full_attn(i):
                x = self._full_attn(i, x, pos, kv, backend)
            else:
                x = self._gdn(i, linear_idx, x, kv, backend)
                linear_idx += 1
            x = self._mlp(i, x, kv, backend)
            if i in aux_layers and hidden_out is not None:
                hidden_out.append(x)
        if hidden_out is not None:
            hidden_out.append(x)
        # lm_head over every prefill position is 4.7% of the FLOPs and a 508 MB
        # output thrown away; the caller passes ``last_only`` (a list gives the
        # per-row valid length of a mixed tick) because a device-side length
        # lookup is a host sync, illegal inside a CUDA graph capture.
        if last_only is not False and x.shape[1] > 1:
            if last_only is True:
                x = autograd.slice(x, ..., slice(x.shape[1] - 1, None), slice(None))
            else:
                idx = torch.as_tensor([n - 1 for n in last_only], device=device)
                x = x[torch.arange(x.shape[0], device=device), idx].unsqueeze(1)
        x = backend.rmsnorm(x, self.params["final_norm"], cfg.rms_eps)
        head_key = "embed_tokens" if cfg.tie_word_embeddings else "lm_head"
        logits = self._linear(backend, x, head_key)
        if getattr(backend, "tp_world", 1) > 1 and not cfg.tie_word_embeddings:
            # Vocab-parallel head; a tied head is the replicated embedding table.
            logits = backend.all_gather(logits, dim=-1)[..., : cfg.vocab_size]
        return logits


def add_lora(
    model: Model, rank: int = 16, alpha: float = 32.0, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Attach LoRA adapters to every linear (quantized or dense bf16) and return
    them. B starts at zero, so step 0 is bit-identical to the base; alpha/rank
    is folded into A's init."""
    g = torch.Generator().manual_seed(seed)
    new: dict[str, torch.Tensor] = {}
    for k in sorted(model.params):
        base, _, suffix = k.rpartition(".")
        if suffix == "wq":
            n, kk = model.params[k].shape[0], model.params[k].shape[1] * 2
        elif suffix == "w8":
            n, kk = model.params[k].shape
        elif (
            model.params[k].ndim == 2
            and not any(k.endswith(x) for x in (".lora_a", ".lora_b"))
            and f"{k}.wq" not in model.params
            and f"{k}.w8" not in model.params
        ):  # dense base: the key IS the weight
            base, n, kk = k, *model.params[k].shape
        else:
            continue
        scale = (alpha / rank) / math.sqrt(kk)
        # On the base weight's device, or materialize() rebuilds the adapter with a new id().
        dev = model.params[k].device
        a = torch.randn(rank, kk, generator=g).to(precision.dtype("adapter")) * scale
        new[base + ".lora_a"] = a.to(dev)
        new[base + ".lora_b"] = torch.zeros(n, rank, dtype=a.dtype, device=dev)
    model.params.update(new)
    return new


def build_random(
    cfg: ModelConfig, seed: int, fuse_projections: bool = False, keep_master: bool = False
) -> Model:
    """Seeded random bf16 model on CPU; fp4 linears are packed from their master,
    which ``keep_master`` retains for the STE backward."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    specs = param_specs(cfg)
    params: dict[str, torch.Tensor] = {}
    norm_keys = ("input_norm", "post_attn_norm", "q_norm", "k_norm", "gdn_norm")
    for key in sorted(specs):
        shape = specs[key]
        if key == "final_norm" or key.endswith(norm_keys):
            params[key] = torch.ones(shape, dtype=torch.bfloat16)
        elif key.endswith(("dt_bias", "a_log")):
            params[key] = torch.zeros(shape, dtype=torch.bfloat16)
        else:
            params[key] = randn(shape)
    if cfg.fp4:
        for key in sorted(fp4_param_keys(cfg)):
            wq, scale = pack_fp4(params[key])
            params[f"{key}.wq"] = wq
            params[f"{key}.scale"], params[f"{key}.oscale"] = renorm_fp4_scale(scale)
            if not keep_master:
                del params[key]
    if fuse_projections:
        _fuse_projections(cfg, params)
    return Model(cfg, params)


# Qwen3_5RMSNorm is zero-centered (y = x_normed * (1 + weight)); the +1 is folded in at load.
_ZERO_CENTERED_NORMS = ("input_norm", "post_attn_norm", "q_norm", "k_norm")

_LAYER_SUFFIXES: dict[str, str] = {
    "input_layernorm.weight": "input_norm",
    "post_attention_layernorm.weight": "post_attn_norm",
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "self_attn.q_norm.weight": "q_norm",
    "self_attn.k_norm.weight": "k_norm",
    "linear_attn.in_proj_qkv.weight": "in_proj_qkv",
    "linear_attn.in_proj_z.weight": "in_proj_z",
    "linear_attn.in_proj_b.weight": "in_proj_b",
    "linear_attn.in_proj_a.weight": "in_proj_a",
    "linear_attn.conv1d.weight": "conv1d",
    "linear_attn.dt_bias": "dt_bias",
    "linear_attn.A_log": "a_log",
    "linear_attn.norm.weight": "gdn_norm",
    "linear_attn.out_proj.weight": "out_proj",
    "mlp.gate_proj.weight": "gate_proj",
    "mlp.up_proj.weight": "up_proj",
    "mlp.down_proj.weight": "down_proj",
}


def _param_key_for(name: str) -> str | None:
    """HF or MLX (prefix and ``.weight`` stripped) tensor name -> param key, or
    None for ignored tensors (vision/MTP/optimizer state)."""
    name = name.removeprefix("model.language_model.").removeprefix("model.")
    top = name.removesuffix(".weight")
    if top == "embed_tokens":
        return "embed_tokens"
    if top == "norm":
        return "final_norm"
    if top == "lm_head":
        return "lm_head"
    if not name.startswith("layers."):
        return None
    layer_str, sep, suffix = name[len("layers.") :].partition(".")
    if not sep or not layer_str.isdigit():
        return None
    mapped = _LAYER_SUFFIXES.get(suffix) or _LAYER_SUFFIXES.get(suffix + ".weight")
    if mapped is None:
        return None
    return f"layers.{int(layer_str)}.{mapped}"


def _is_lm_head(name: str) -> bool:
    stem = name.removeprefix("model.language_model.").removeprefix("model.")
    stem = stem.removeprefix("language_model.")
    return stem == "lm_head" or stem.startswith("lm_head.")


def _validate_hf_config(cfg: ModelConfig, hf_cfg: dict, source: str) -> None:
    # Qwen3.5 wraps the text model in text_config; flat Qwen3 exports do not.
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    checks = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_layers,
        "vocab_size": cfg.vocab_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
    }
    for field, expected in checks.items():
        actual = text_cfg.get(field)
        if actual is not None and actual != expected:
            raise RuntimeError(
                f"config mismatch for `{source}`: {field}={actual} in HF config.json "
                f"but cfg expects {expected}"
            )
    layer_types = text_cfg.get("layer_types")
    if layer_types is not None:
        full = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        if full != tuple(cfg.full_attn_layers):
            raise RuntimeError(
                f"config mismatch for `{source}`: HF layer_types has full-attention "
                f"layers {full} but cfg.full_attn_layers is {tuple(cfg.full_attn_layers)}"
            )
    if "attn_output_gate" in text_cfg and text_cfg["attn_output_gate"] != cfg.full_attn_gated:
        raise RuntimeError(
            f"config mismatch for `{source}`: HF attn_output_gate="
            f"{text_cfg['attn_output_gate']} but cfg.full_attn_gated={cfg.full_attn_gated}"
        )
    # Qwen3.5 nests RoPE under `rope_parameters`; flat Qwen3 exports keep it top level.
    rope = text_cfg.get("rope_parameters") or {}
    theta = rope.get("rope_theta", text_cfg.get("rope_theta"))
    if theta is not None and float(theta) != float(cfg.rope_theta):
        raise RuntimeError(
            f"config mismatch for `{source}`: rope_theta={theta} in HF config.json "
            f"but cfg expects {cfg.rope_theta}"
        )
    prf = rope.get("partial_rotary_factor", text_cfg.get("partial_rotary_factor"))
    if prf is not None:
        rd = float(prf) * cfg.head_dim  # rounded: rotary_dim is an even integer
        if round(rd) != cfg.effective_rotary_dim:
            raise RuntimeError(
                f"config mismatch for `{source}`: partial_rotary_factor={prf} gives "
                f"rotary_dim {rd} at head_dim {cfg.head_dim} but cfg expects "
                f"{cfg.effective_rotary_dim}"
            )
    # Only unscaled RoPE is implemented; serving YaRN/linear unscaled is wrong at every position.
    scaling = text_cfg.get("rope_scaling") or {}
    rope_type = scaling.get("rope_type") or scaling.get("type") or rope.get("rope_type")
    if rope_type not in (None, "default") or scaling.get("factor", 1.0) != 1.0:
        raise RuntimeError(
            f"`{source}`: checkpoint declares RoPE scaling (rope_scaling={scaling!r}, "
            f"rope_parameters.rope_type={rope_type!r}) and tileRL implements none — "
            f"it would be served as unscaled RoPE, silently wrong at every position"
        )
    # Multimodal configs carry tie_word_embeddings at the top level, overriding text_config.
    tie = hf_cfg.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings"))
    if tie is not None and bool(tie) != cfg.tie_word_embeddings:
        raise RuntimeError(
            f"config mismatch for `{source}`: tie_word_embeddings={tie} in HF "
            f"config.json but cfg expects {cfg.tie_word_embeddings}"
        )


def _shard_files(ckpt_dir: Path, source: str) -> list[Path]:
    index = ckpt_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        weight_map = data.get("weight_map", {})
        if not weight_map:
            raise RuntimeError(f"`{source}`: empty weight_map in {index}")
        names = sorted(set(weight_map.values()))
        files = [ckpt_dir / name for name in names]
    else:
        shards = sorted(ckpt_dir.glob("model-*.safetensors"))
        files = shards if shards else [ckpt_dir / "model.safetensors"]
    missing = [f for f in files if not f.exists()]
    if missing:
        tried = index if index.exists() else ckpt_dir
        raise RuntimeError(
            f"`{source}`: no safetensors shards found (looked for {files}; resolved via {tried})"
        )
    return files


def _dequant_mlx(
    w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int
) -> torch.Tensor:
    """MLX affine 4-bit: ``w`` uint32 [out, in//8] low nibble first, ``w = s*q + b`` per group."""
    out, k8 = w.shape
    shifts = torch.arange(8, dtype=torch.int64) * 4
    q = ((w.long().unsqueeze(-1) >> shifts) & 0xF).float().reshape(out, k8 * 8)
    s = scales.float().repeat_interleave(group_size, dim=-1)
    b = biases.float().repeat_interleave(group_size, dim=-1)
    return (s * q + b).to(torch.bfloat16)


def load_hf(
    cfg: ModelConfig,
    source: str,
    num_layers: int | None = None,
    fuse_projections: bool = False,
    keep_master: bool = False,
) -> Model:
    """Load ``source`` (HF repo id or local directory) into a Model; every
    failure raises RuntimeError. Accepts bf16, MLX 4-bit, ModelOpt/official
    NVFP4 (served as its own bytes), FP8 block/per-tensor (kept native as
    ``.w8/.wscale``), AWQ-int4 and :func:`save_hf` output. ``num_layers``
    truncates; ``keep_master`` (training) regenerates the bf16 STE master."""
    if num_layers is not None and not 0 < num_layers <= cfg.num_layers:
        raise ValueError(f"num_layers={num_layers} out of range for {cfg.num_layers} layers")
    src = Path(source)
    if src.is_dir():
        ckpt_dir = src
        source_desc = str(src)
    else:
        from huggingface_hub import snapshot_download

        try:
            ckpt_dir = Path(
                snapshot_download(
                    source,
                    allow_patterns=["config.json", "model.safetensors*", "model-*.safetensors"],
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to download HF repo `{source}` via huggingface_hub: {exc}"
            ) from exc
        source_desc = source

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"`{source_desc}`: config.json not found at {config_path}")
    hf_cfg = json.loads(config_path.read_text())
    _validate_hf_config(cfg, hf_cfg, source_desc)
    if num_layers is not None:  # after validation: the checkpoint is the full model
        cfg = replace(
            cfg,
            num_layers=num_layers,
            full_attn_layers=tuple(i for i in cfg.full_attn_layers if i < num_layers),
        )
        print(
            f"[tilerl.model] num_layers={num_layers}: loading embedding + "
            f"first {num_layers} layers + final norm"
            + ("" if cfg.tie_word_embeddings else " + lm_head")
        )

    from safetensors.torch import load_file

    specs = param_specs(cfg)
    group_size = hf_cfg.get("quantization", {}).get("group_size", 64)
    awq_group = (hf_cfg.get("quantization_config") or {}).get("group_size", 128)
    params: dict[str, torch.Tensor] = {}
    fp8_native: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
    fp4_native: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    lm_head_tensor: str | None = None
    for shard in _shard_files(ckpt_dir, source_desc):
        tensors = load_file(str(shard))
        lm_head_tensor = lm_head_tensor or next((n for n in tensors if _is_lm_head(n)), None)
        mlx = next((n for n in tensors if n.startswith("language_model.")), None) is not None
        if mlx:
            tensors = {n[len("language_model.") :]: t for n, t in tensors.items()}
        for hf_name, tensor in tensors.items():
            if mlx:
                if hf_name.endswith((".scales", ".biases")):
                    continue  # consumed with the .weight sibling below
                base = hf_name[: -len(".weight")] if hf_name.endswith(".weight") else hf_name
                if base + ".scales" in tensors:
                    tensor = _dequant_mlx(
                        tensor, tensors[base + ".scales"], tensors[base + ".biases"], group_size
                    )
                key = _param_key_for(base)
            elif hf_name.endswith(".weight_packed"):  # ModelOpt NVFP4
                stem = hf_name.removesuffix(".weight_packed")
                key = _param_key_for(stem + ".weight")
                if key is not None:
                    sib = (tensors[stem + ".weight_scale"], tensors[stem + ".weight_global_scale"])
                    if cfg.fp4:
                        fp4_native[key] = _native_fp4(tensor, *sib, divide=True)
                        key = None  # served packed
                    else:
                        tensor = dequant_nvfp4(tensor, *sib, global_divide=True)
            elif hf_name.endswith(".wq"):  # save_hf output: the served bytes verbatim
                stem = hf_name.removesuffix(".wq")
                key = _param_key_for(stem + ".weight")
                if key is not None:
                    sib = (tensors[stem + ".scale"].float(), tensors[stem + ".oscale"].float())
                    if cfg.fp4:
                        fp4_native[key] = (tensor, *sib)
                        key = None
                    else:
                        tensor = unpack_fp4(tensor, *sib)
            elif hf_name.endswith(".qweight"):  # AWQ-int4
                stem = hf_name.removesuffix(".qweight")
                key = _param_key_for(stem + ".weight")
                if key is not None:
                    tensor = dequant_awq(
                        tensor, tensors[stem + ".scales"], tensors[stem + ".qzeros"], awq_group
                    )
            elif (
                hf_name.endswith(".weight")
                and hf_name.removesuffix(".weight") + ".weight_scale_inv" in tensors
            ):
                # ModelOpt FP8 block; "scale_inv" is multiplied despite the name.
                key = _param_key_for(hf_name)
                if key is not None:
                    wscale = tensors[hf_name.removesuffix(".weight") + ".weight_scale_inv"].float()
                    fp8_native[key] = (tensor, wscale, None)
                    tensor = dequant_fp8(tensor, wscale).to(torch.bfloat16)
            elif (
                hf_name.endswith(".weight")
                and hf_name.removesuffix(".weight") + ".weight_scale_2" in tensors
            ):
                # Official NVFP4: ModelOpt's math under the official tensor names.
                stem = hf_name.removesuffix(".weight")
                key = _param_key_for(hf_name)
                if key is not None:
                    sib = (tensors[stem + ".weight_scale"], tensors[stem + ".weight_scale_2"])
                    if cfg.fp4:
                        fp4_native[key] = _native_fp4(tensor, *sib)
                        key = None
                    else:
                        tensor = dequant_nvfp4(tensor, *sib)
            elif (
                hf_name.endswith(".weight")
                and hf_name.removesuffix(".weight") + ".weight_scale" in tensors
            ):
                # FP8 with a per-tensor or per-channel scale: both are per-row
                # constants, so they ride .oscale over a ones wscale.
                stem = hf_name.removesuffix(".weight")
                key = _param_key_for(hf_name)
                if key is not None:
                    n, k = tensor.shape
                    ws = tensors[stem + ".weight_scale"].float().reshape(-1)
                    oscale = ws.expand(n).contiguous()
                    wscale = torch.ones(((n + 127) // 128), ((k + 127) // 128))
                    fp8_native[key] = (tensor, wscale, oscale)
                    # the master carries the full magnitude, oscale included
                    tensor = (tensor.float() * oscale.reshape(-1, 1)).to(torch.bfloat16)
            elif hf_name.endswith(
                (
                    ".weight_scale",
                    ".weight_scale_2",
                    ".weight_global_scale",
                    ".input_global_scale",
                    ".input_scale",
                    ".weight_scale_inv",
                    ".scales",
                    ".qzeros",
                )
            ):
                continue  # consumed with the weight above; input_* is activation quant
            else:
                key = _param_key_for(hf_name)
            if key is None or (num_layers is not None and key not in specs):
                continue
            if key in params:
                raise RuntimeError(
                    f"`{source_desc}`: tensor for param `{key}` appears in multiple "
                    f"shards (last: {shard.name})"
                )
            if key.endswith(".conv1d") and tensor.ndim == 3:
                tensor = tensor.reshape(tensor.shape[0], -1)  # [C,1,K]/[C,K,1] -> [C,K]
            if key == "final_norm" or key.endswith(_ZERO_CENTERED_NORMS):
                tensor = (tensor.float() + 1.0).to(tensor.dtype)
            params[key] = tensor

    for key, (w8, wscale, oscale) in fp8_native.items():
        params[f"{key}.w8"] = w8.contiguous()
        params[f"{key}.wscale"] = wscale.contiguous()
        if oscale is not None:
            params[f"{key}.oscale"] = oscale

    # Validated here: the shape check below only sees keys with a bf16 tensor.
    for key, (wq, scale, oscale) in fp4_native.items():
        if key not in specs:
            continue  # truncated-away layer
        n, k = specs[key]
        if tuple(wq.shape) != (n, k // 2) or scale.shape[0] != n or k % scale.shape[1]:
            raise RuntimeError(
                f"`{source_desc}`: packed `{key}` is {tuple(wq.shape)} / "
                f"{tuple(scale.shape)}, expected ({n}, {k // 2}) and ({n}, K/B)"
            )
        params[f"{key}.wq"], params[f"{key}.scale"], params[f"{key}.oscale"] = wq, scale, oscale
        if keep_master:
            params[key] = unpack_fp4(wq, scale, oscale)

    # Tied cfg + a shipped lm_head would silently serve embed_tokens as the head.
    if cfg.tie_word_embeddings:
        if lm_head_tensor is not None:
            raise RuntimeError(
                f"`{source_desc}`: cfg.tie_word_embeddings=True but the checkpoint "
                f"ships `{lm_head_tensor}` — set tie_word_embeddings=False, or the "
                f"embedding is served as the output projection, silently wrong"
            )
    elif "lm_head" not in params and not _quantized(params, "lm_head"):
        raise RuntimeError(f"`{source_desc}`: untied model is missing lm_head.weight")

    missing = sorted(k for k in specs if not (k in params or _quantized(params, k)))
    if missing:
        raise RuntimeError(
            f"`{source_desc}`: checkpoint is missing {len(missing)} expected "
            f"tensors, e.g. {missing[:5]}"
        )
    for key, tensor in params.items():
        if key in specs and tuple(tensor.shape) != specs[key]:
            raise RuntimeError(
                f"`{source_desc}`: shape mismatch for `{key}`: checkpoint "
                f"{tuple(tensor.shape)} vs cfg {specs[key]}"
            )

    # Pack the bf16 linears the checkpoint did not ship quantized.
    if cfg.fp4:
        for key in sorted(fp4_param_keys(cfg)):
            if _quantized(params, key):
                continue
            master = params[key]
            if master.dtype != torch.bfloat16:
                master = master.to(torch.bfloat16)
                params[key] = master
            wq, scale = pack_fp4(master)
            params[f"{key}.wq"] = wq
            params[f"{key}.scale"], params[f"{key}.oscale"] = renorm_fp4_scale(scale)

    if not keep_master:  # serving: the quantized bytes ARE the weight (embedding needs its table)
        for key in [k for k in specs if k != "embed_tokens" and _quantized(params, k)]:
            params.pop(key, None)

    if fuse_projections:
        _fuse_projections(cfg, params)
    return Model(cfg, params)


_HF_SUFFIXES = {v: k for k, v in _LAYER_SUFFIXES.items()}


def _hf_tensor_name(key: str) -> str:
    if key == "embed_tokens":
        return "model.language_model.embed_tokens.weight"
    if key == "final_norm":
        return "model.language_model.norm.weight"
    if key == "lm_head":
        return "lm_head.weight"
    _, layer, suffix = key.split(".", 2)
    return f"model.language_model.layers.{layer}.{_HF_SUFFIXES[suffix]}"


def drop_quantized(model: Model) -> Model:
    """Free the served bytes of every linear with a bf16 master: full fine-tuning
    never reads them (14.9 GiB on the 27B) and ``save_hf`` re-packs from the master."""
    for key in [k for k in param_specs(model.cfg) if k in model.params]:
        for suffix in (".wq", ".scale", ".oscale", ".w8", ".wscale"):
            model.params.pop(key + suffix, None)
    return model


def save_hf(model: Model, path: str | Path) -> None:
    """HF safetensors + config.json: a bf16 master is saved as the weight, else
    the served fp4 bytes verbatim, so ``load_hf(save_hf(m))`` is bit-identical.
    # ponytail: no optimizer state; training-state checkpoints are day-2."""
    from dataclasses import asdict

    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    tensors: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for key in param_specs(cfg):
        hf = _hf_tensor_name(key)
        if key in model.params:
            t = model.params[key]
            if key == "final_norm" or key.endswith(_ZERO_CENTERED_NORMS):
                t = (t.float() - 1.0).to(t.dtype)  # undo load's zero-centered +1 fold
            tensors[hf] = t
        elif f"{key}.wq" in model.params:
            stem = hf.removesuffix(".weight")
            for suffix in (".wq", ".scale", ".oscale"):
                t = model.params[key + suffix]
                if getattr(t, "_tl_twiddled", False):  # sm90 serves twiddled bytes
                    t = untwiddle_fp4(t)
                tensors[stem + suffix] = t
        else:
            missing.append(key)
    if missing:
        raise RuntimeError(
            f"save_hf: {len(missing)} params carry neither a bf16 master nor fp4 "
            f"bytes (fused-projection or native-fp8 serving model), e.g. {missing[:3]}"
        )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
        str(path / "model.safetensors"),
    )
    layer_types = [
        "full_attention" if i in cfg.full_attn_layer_set else "linear_attention"
        for i in range(cfg.num_layers)
    ]
    config = asdict(cfg) | {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "num_hidden_layers": cfg.num_layers,
        "num_key_value_heads": cfg.num_kv_heads,
        "attn_output_gate": cfg.full_attn_gated,
        "layer_types": layer_types,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2))
