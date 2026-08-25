"""Qwen3.5/3.6 hybrid model: full-attention + gated-delta layers, plus
checkpoint loading (HF safetensors and MLX 4-bit affine).

Layer order and residuals follow agent-infer's qwen35 forward
(``crates/infer-cuda/src/qwen35_forward.rs``)::

    x = embed(input_ids)
    for layer i:
        x = x + attn(rmsnorm(input_norm, x))      # full-attn or gated-delta
        x = x + mlp(rmsnorm(post_attn_norm, x))   # silu(gate) * up -> down
    logits = lm_head(rmsnorm(final_norm, x))      # tied to embed when cfg says so

Backend op contract (this module calls only backend ops, never tilelang/torch
math). Beyond the pinned op list, the architecture forces four extensions:

* ``add(a, b)`` — residual adds (the pinned list has no elementwise add).
* ``rope(x, positions, theta, rotary_dim=None)`` — partial RoPE (Qwen3.8:
  rotary_dim=64 of head_dim=256); defaults to head_dim.
* ``paged_attention(q, k_cache, v_cache, block_table, seq_lens, scale,
  gate=None)`` — with ``gate`` (the q_proj gate half, [B,T,Hq,D]) returns
  ``attn_out * sigmoid(gate)`` (full-attn output gate, BEFORE o_proj).
* ``linear_attn_chunk`` / ``linear_attn_step(q, k, v, g, beta, state, *, z,
  conv1d_weight, dt_bias, a_log, norm_weight, conv_window=None)`` — the FULL
  gated-delta layer core, mirroring agent-infer's host reference
  ``linear_attention_forward`` (autograd/src/ops/linear_attention.rs:2385):
  causal depthwise conv1d over the raw qkv -> SiLU -> q L2-norm /sqrt(K),
  k L2-norm -> beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt) -> delta-rule
  recurrence over the f32 state [B,H,K,V] -> RMSNorm(norm_weight) -> *silu(z).
  ``q/k/v`` are the raw in_proj_qkv slices (pre-conv), ``g``/``beta`` the raw
  in_proj_a/in_proj_b outputs. Returns ``(out [B,T,v_dim], new_state,
  new_window)`` (value heads flattened into the trailing dim).
  ``conv_window`` [B,K-1,qkv_dim] carries the previous segment's raw qkv so
  segmented decode is exact; None -> zero-padded one-shot prefill (then
  ``new_window`` is None).

BatchKv assumptions (Engine attaches the pools at submit): ``kv.kv_pool`` /
``kv.state_pool``; ``k_pool``/``v_pool`` are ``[num_layers, num_blocks,
num_kv_heads, BLOCK_TOKENS, head_dim]``; ``backend.write_tokens(k, v, kv,
layer_idx)`` scatters ``k/v`` [B,T,Hkv,D] at ``[seq_len-T, seq_len)`` through
the block table (one capturable kernel on sm90, the pool's torch loop on other
arches); ``state_pool.states`` is ``[num_slots, num_linear_layers, H, K, V]``
and ``state_pool.conv_windows`` ``[num_slots, num_linear_layers, K-1,
qkv_dim]`` (None without GDN layers).

Checkpoint loading: ``load_hf(cfg, source, ...)`` maps a Qwen3.5/3.6
safetensors checkpoint into the param dict (``build_random`` draws the same
schema). Tensor names follow agent-infer's ``qwen35-spec``
``layer_tensor_names`` (``crates/qwen35-spec/src/lib.rs``):
``model.language_model.embed_tokens.weight``->``embed_tokens``,
``...norm.weight``->``final_norm``, ``layers.{i}.input_layernorm.weight``->
``input_norm``, ``...post_attention_layernorm.weight``->``post_attn_norm``;
full-attn ``self_attn.{q,k,v,o}_proj.weight`` + ``q_norm/k_norm.weight``;
gated-delta ``linear_attn.in_proj_{qkv,z,b,a}.weight``, ``conv1d.weight``,
``dt_bias``, ``A_log``, ``norm.weight``, ``out_proj.weight``; MLP
``mlp.{gate,up,down}_proj.weight``; ``lm_head.weight`` only when untied.
``layer_types`` from HF ``config.json`` is read verbatim and validated against
``cfg.full_attn_layers`` — indices are never hardcoded. fp4 linears are
quantized on load when ``cfg.fp4``. Every failure raises with the exact source
and files tried; this module never fakes success.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from . import autograd
from .config import ModelConfig, tiny
from .ops.reference import (
    dequant_awq,
    dequant_fp8,
    dequant_nvfp4,
    pack_fp4,
    unpack_fp4,
)  # re-exported for callers

if TYPE_CHECKING:  # pragma: no cover - typing only, no tilelang import at runtime
    import numpy as np

    from .ops.backend import Backend


# --- Param schema -----------------------------------------------------------
def param_specs(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Canonical param keys + shapes for ``cfg`` (single source of truth:
    ``build_random`` draws these, ``load_hf`` validates against them)."""
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
    """Keys of the 2D projection matrices that are fp4-packed when cfg.fp4.

    Embeddings, norms, conv1d (K=4 < block 16), dt_bias and a_log stay bf16.
    """
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


def _projection_groups(cfg: ModelConfig, layer_idx: int) -> list[tuple[str, list[str]]]:
    """Same-input fp4 projection groups, fused at load for serving decode.

    Each group reads the same post-norm hidden and is fp4-packed, so the
    packed weights concat losslessly along N (per-32-block scales are
    per-row) and one GEMV replaces the group's launches — the decode tick is
    launch-latency-bound on the small projections (N=48/1024 at 0.1-3% roof),
    so collapsing launches is the lever. Serving-only: training keeps the
    unfused masters (the fused key has none, so its tape backward would have
    nowhere to land the STE grad).
    """
    p = f"layers.{layer_idx}"
    groups = [(f"{p}.gate_up", [f"{p}.gate_proj", f"{p}.up_proj"])]
    if cfg.is_full_attn(layer_idx):
        groups.append((f"{p}.qkv", [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"]))
    else:
        groups.append((f"{p}.ab", [f"{p}.in_proj_a", f"{p}.in_proj_b"]))
    return groups


def _fuse_projections(cfg: ModelConfig, params: dict[str, torch.Tensor]) -> None:
    """Concat each group's packed fp4 weights into a fused key (in-place)."""
    for i in range(cfg.num_layers):
        for fused_key, group in _projection_groups(cfg, i):
            if f"{fused_key}.wq" in params:
                continue
            try:
                wqs = [params[f"{k}.wq"] for k in group]
                scales = [params[f"{k}.scale"] for k in group]
            except KeyError:
                continue  # group not fully fp4 (bf16 checkpoint) — skip
            params[f"{fused_key}.wq"] = torch.cat(wqs, dim=0).contiguous()
            params[f"{fused_key}.scale"] = torch.cat(scales, dim=0).contiguous()
            for k in group:  # drop the dead copies (bf16 masters stay, recording-only)
                del params[f"{k}.wq"]
                del params[f"{k}.scale"]


# --- Model ------------------------------------------------------------------
class Model:
    """Qwen3.5/3.6 hybrid model. ``params`` maps :func:`param_specs` keys to
    bf16 tensors; fp4 linears also carry ``<key>.wq`` (uint8) and
    ``<key>.scale`` (f32) alongside the bf16 master, and native-fp8 linears
    carry ``<key>.w8`` (e4m3) and ``<key>.wscale`` (f32 per-128-block)."""

    def __init__(self, cfg: ModelConfig, params: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.params = params

    # -- linear dispatch (fp4-packed / native-fp8 when present, plain bf16 otherwise) ----
    def _linear(self, backend: "Backend", x: torch.Tensor, key: str) -> torch.Tensor:
        wq = self.params.get(key + ".wq")
        if wq is not None:
            # ``master`` is recording-only: the STE grad lands on the bf16
            # master weight (see autograd._linear_fp4). Fused projection keys
            # (serving-only) have no master — the tape never sees them.
            return backend.linear_fp4(
                x, wq, self.params[key + ".scale"], master=self.params.get(key)
            )
        w8 = self.params.get(key + ".w8")
        if w8 is not None:
            # Native fp8: the sm90 prefill path computes with w8 directly;
            # the bf16 master is recording-only (STE grad, see
            # autograd._linear_fp8) and the decode (M=1) fallback.
            return backend.linear_fp8(x, w8, self.params[key + ".wscale"], master=self.params[key])
        return backend.linear(x, self.params[key])

    # -- full attention layer ----------------------------------------------
    def _full_attn(
        self,
        layer_idx: int,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv: Any,
        backend: "Backend",
    ) -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.input_norm"], cfg.rms_eps)
        hq, hkv, d = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        qkv_key = f"{p}.qkv"
        if f"{qkv_key}.wq" in self.params:  # fused q/k/v (serving)
            qkv = self._linear(backend, h, qkv_key)
            q_rows = hq * d * (2 if cfg.full_attn_gated else 1)
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
        # Per-head RMSNorm (weight [head_dim], broadcast over B,T,H).
        q = backend.rmsnorm(q, self.params[f"{p}.q_norm"], cfg.rms_eps)
        k = backend.rmsnorm(k, self.params[f"{p}.k_norm"], cfg.rms_eps)
        q = backend.rope(q, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        k = backend.rope(k, positions, cfg.rope_theta, rotary_dim=cfg.effective_rotary_dim)
        if getattr(kv, "dense", False):
            # Training: dense GQA attention (no paged pool indirection).
            out = backend.attention(q, k, v, 1.0 / math.sqrt(d), gate=gate)
        else:
            backend.write_tokens(k, v, kv, layer_idx)
            out = backend.paged_attention(
                q,
                kv.kv_pool.k_pool[layer_idx],
                kv.kv_pool.v_pool[layer_idx],
                kv.block_table,
                kv.seq_len,
                1.0 / math.sqrt(d),
                gate=gate,
            )
        out = self._linear(backend, autograd.reshape(out, b, t, hq * d), f"{p}.o_proj")
        return backend.add(x, out)

    # -- gated-delta (linear attention) layer -------------------------------
    def _gdn(
        self,
        layer_idx: int,
        linear_idx: int,
        x: torch.Tensor,
        kv: Any,
        backend: "Backend",
    ) -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.input_norm"], cfg.rms_eps)
        qkv = self._linear(backend, h, f"{p}.in_proj_qkv")
        z = self._linear(backend, h, f"{p}.in_proj_z")
        ab_key = f"{p}.ab"
        if f"{ab_key}.wq" in self.params:  # fused a/b (serving)
            ab = self._linear(backend, h, ab_key)
            nvh = cfg.linear_num_value_heads
            a_proj = autograd.slice(ab, ..., slice(0, nvh))
            b_proj = autograd.slice(ab, ..., slice(nvh, None))
        else:
            b_proj = self._linear(backend, h, f"{p}.in_proj_b")
            a_proj = self._linear(backend, h, f"{p}.in_proj_a")
        qd, kd = cfg.linear_q_dim, cfg.linear_k_dim
        # Recorded slices of the fused projection: the tape must see the split
        # or the grad never reaches in_proj_qkv (views break the id() chain).
        q = autograd.slice(qkv, ..., slice(0, qd))
        k = autograd.slice(qkv, ..., slice(qd, qd + kd))
        v = autograd.slice(qkv, ..., slice(qd + kd, None))
        slots = torch.as_tensor(kv.state_slot, dtype=torch.long).reshape(-1)
        state = kv.state_pool.states[slots, linear_idx]  # [B,H,K,V]
        window = (
            kv.state_pool.conv_windows[slots, linear_idx]
            if kv.state_pool.conv_windows is not None
            else None
        )
        kwargs = dict(
            z=z,
            conv1d_weight=self.params[f"{p}.conv1d"],
            dt_bias=self.params[f"{p}.dt_bias"],
            a_log=self.params[f"{p}.a_log"],
            norm_weight=self.params[f"{p}.gdn_norm"],
            conv_window=window,
        )
        out, new_state, new_window = backend.linear_attn_chunk(
            q, k, v, a_proj, b_proj, state, **kwargs
        )
        kv.state_pool.states[slots, linear_idx] = new_state.to(kv.state_pool.states.dtype)
        if new_window is not None:
            kv.state_pool.conv_windows[slots, linear_idx] = new_window.to(
                kv.state_pool.conv_windows.dtype
            )
        out = self._linear(backend, out, f"{p}.out_proj")
        return backend.add(x, out)

    # -- MLP ----------------------------------------------------------------
    def _mlp(self, layer_idx: int, x: torch.Tensor, backend: "Backend") -> torch.Tensor:
        cfg = self.cfg
        p = f"layers.{layer_idx}"
        h = backend.rmsnorm(x, self.params[f"{p}.post_attn_norm"], cfg.rms_eps)
        gu_key = f"{p}.gate_up"
        if f"{gu_key}.wq" in self.params:  # fused gate/up (serving)
            gu = self._linear(backend, h, gu_key)
            gate = autograd.slice(gu, ..., slice(0, cfg.intermediate_size))
            up = autograd.slice(gu, ..., slice(cfg.intermediate_size, None))
        else:
            gate = self._linear(backend, h, f"{p}.gate_proj")
            up = self._linear(backend, h, f"{p}.up_proj")
        activated = backend.silu_mul(gate, up)
        down = self._linear(backend, activated, f"{p}.down_proj")
        return backend.add(x, down)

    # -- full forward --------------------------------------------------------
    def forward(
        self,
        input_ids: "np.ndarray | torch.Tensor",
        positions: "np.ndarray | torch.Tensor",
        kv: Any,
        backend: "Backend",
    ) -> torch.Tensor:
        """Run the model. ``input_ids`` [B,T] int, ``positions`` [T] or [B,T]
        int (RoPE positions), ``kv`` a BatchKv with pools attached. Returns
        logits [B,T,vocab_size] on the backend device."""
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
            x = self._mlp(i, x, backend)
        x = backend.rmsnorm(x, self.params["final_norm"], cfg.rms_eps)
        head_key = "embed_tokens" if cfg.tie_word_embeddings else "lm_head"
        return self._linear(backend, x, head_key)


# --- Random initialization --------------------------------------------------
def build_random(cfg: ModelConfig, seed: int, fuse_projections: bool = False) -> Model:
    """Deterministic random model: N(0, 0.02^2) matrices, ones for norms,
    zeros for dt_bias/a_log, all bf16 on CPU. fp4 linears are packed from
    their bf16 master (master kept for the STE backward)."""
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
            params[f"{key}.scale"] = scale
        # ponytail: fp4 masters double the weight memory (54GB for 27B bf16);
        # inference-only runs could drop them, training needs them for the STE.
    if fuse_projections:
        _fuse_projections(cfg, params)
    return Model(cfg, params)


# --- Checkpoint loading -----------------------------------------------------

#: HF suffix -> param key (layer index substituted by the caller).
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
    """Map one HF or MLX tensor name to a param key, or None for tensors we
    ignore (vision/MTP/optimizer state). HF names keep ``model.language_model.``
    and ``.weight``; MLX names arrive with both stripped (the suffix table
    lookup tries the bare key and ``key + ".weight"``)."""
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


def _validate_hf_config(cfg: ModelConfig, text_cfg: dict, source: str) -> None:
    """Cross-check the HF config.json against ``cfg``. Mismatches raise —
    loading a checkpoint into the wrong-shaped model must fail loudly."""
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
    # layer_types is the verbatim source of truth for the full/linear split.
    layer_types = text_cfg.get("layer_types")
    if layer_types is not None:
        full = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        if full != tuple(cfg.full_attn_layers):
            raise RuntimeError(
                f"config mismatch for `{source}`: HF layer_types has full-attention "
                f"layers {full} but cfg.full_attn_layers is {tuple(cfg.full_attn_layers)}"
            )
    # The gated q_proj changes q_proj's row count — catch it before loading.
    if "attn_output_gate" in text_cfg and text_cfg["attn_output_gate"] != cfg.full_attn_gated:
        raise RuntimeError(
            f"config mismatch for `{source}`: HF attn_output_gate="
            f"{text_cfg['attn_output_gate']} but cfg.full_attn_gated={cfg.full_attn_gated}"
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
    if missing or not files:
        tried = index if index.exists() else ckpt_dir
        raise RuntimeError(
            f"`{source}`: no safetensors shards found (looked for {files}; resolved via {tried})"
        )
    return files


def _dequant_mlx(
    w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int
) -> torch.Tensor:
    """MLX affine 4-bit dequant. ``w`` uint32 [out, in//8] (8 nibbles/uint32,
    low bits first), ``scales``/``biases`` [out, in//group]; ``w = s*q + b``
    per group (the MLX quantized-matmul kernel convention — scales may be
    negative). Returns bf16 [out, in]."""
    out, k8 = w.shape
    shifts = torch.arange(8, dtype=torch.int64) * 4
    q = ((w.long().unsqueeze(-1) >> shifts) & 0xF).float().reshape(out, k8 * 8)
    s = scales.float().repeat_interleave(group_size, dim=-1)
    b = biases.float().repeat_interleave(group_size, dim=-1)
    return (s * q + b).to(torch.bfloat16)


def load_hf(
    cfg: ModelConfig, source: str, num_layers: int | None = None, fuse_projections: bool = False
) -> Model:
    """Load ``source`` (HF repo id or local checkpoint directory) into a Model.

    ``num_layers`` truncates to the first N layers (embedding + N layers +
    final norm + lm_head); tensors for the skipped layers are not required.
    Raises RuntimeError (never a silent fallback) on download failure, missing
    files, config mismatch, or missing/duplicate/shape-mismatched tensors.
    HF safetensors and MLX 4-bit affine checkpoints are both accepted (the
    MLX path is detected from the ``language_model.`` tensor-name prefix), as
    are ModelOpt NVFP4/FP8-block checkpoints (detected from the
    ``weight_packed`` / ``weight_scale_inv`` sibling tensors), official-NVFP4
    checkpoints (``weight_scale_2`` sibling: e2m1 nibbles * f8 block scale *
    global scale), per-tensor FP8 (f8 ``weight`` + scalar ``weight_scale``),
    and AWQ-int4 (``qweight`` / ``scales`` / ``qzeros`` siblings, group size
    from ``quantization_config.group_size``). FP8 linears are kept native:
    the e4m3 weight lands in ``<key>.w8`` and the per-128-block scale in
    ``<key>.wscale`` (a per-tensor scalar is expanded to the same layout),
    with the bf16 dequant kept as the recording-only master."""
    if num_layers is not None and not 0 < num_layers <= cfg.num_layers:
        raise ValueError(f"num_layers={num_layers} out of range for {cfg.num_layers} layers")
    src = Path(source)
    if src.is_dir():
        ckpt_dir = src
        source_desc = str(src)
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dep is pinned
            raise RuntimeError(
                f"huggingface-hub is required to load `{source}` (pip install huggingface-hub)"
            ) from exc
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
    # Qwen3.5 wraps the text model in text_config; flat Qwen3 exports do not.
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    _validate_hf_config(cfg, text_cfg, source_desc)
    if num_layers is not None:
        # Truncate AFTER validation: the checkpoint is the full model; only
        # the loaded tensor set and the returned config are truncated.
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

    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - dep is pinned
        raise RuntimeError(
            f"safetensors is required to load `{source_desc}` (pip install safetensors)"
        ) from exc

    specs = param_specs(cfg)
    group_size = hf_cfg.get("quantization", {}).get("group_size", 64)
    awq_group = (hf_cfg.get("quantization_config") or {}).get("group_size", 128)
    params: dict[str, torch.Tensor] = {}
    #: Native-fp8 linears (key -> (e4m3 weight, f32 per-128-block scale)),
    #: kept native instead of dequantized: the bf16 master below is recording-
    #: only, the sm90 prefill path computes with the e4m3 weight directly.
    fp8_native: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for shard in _shard_files(ckpt_dir, source_desc):
        tensors = load_file(str(shard))
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
            elif hf_name.endswith(".weight_packed"):
                # ModelOpt NVFP4 (Qwen3.6 MLP linears): packed e2m1 nibbles +
                # f8 block scale + global scale siblings, dequantized to bf16.
                # The stored global scale is its reciprocal (divide, not
                # multiply — agent-infer quant_format.rs ScaleApply::Divide).
                stem = hf_name.removesuffix(".weight_packed")
                key = _param_key_for(stem + ".weight")
                if key is not None:
                    tensor = dequant_nvfp4(
                        tensor,
                        tensors[stem + ".weight_scale"],
                        tensors[stem + ".weight_global_scale"],
                        global_divide=True,
                    )
            elif hf_name.endswith(".qweight"):
                # AWQ-int4 (autoawq GEMM): packed int4 weights + per-group
                # scales/zeros siblings, dequantized to bf16.
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
                # ModelOpt FP8 block (Qwen3.6 GDN linears): f8 weight +
                # per-128-block scales, kept native (the bf16 dequant is the
                # recording-only master). The stored "scale_inv" is the scale
                # itself (multiplied, despite the name — agent-infer
                # quant_format.rs ScaleApply::Multiply).
                key = _param_key_for(hf_name)
                if key is not None:
                    wscale = tensors[hf_name.removesuffix(".weight") + ".weight_scale_inv"].float()
                    fp8_native[key] = (tensor, wscale)
                    tensor = dequant_fp8(tensor, wscale).to(torch.bfloat16)
            elif (
                hf_name.endswith(".weight")
                and hf_name.removesuffix(".weight") + ".weight_scale_2" in tensors
            ):
                # Official NVFP4 (nvidia/Qwen3.6-27B-NVFP4 MLP linears): same
                # e2m1*f8-block-scale*global-scale math as ModelOpt, official
                # tensor names (weight / weight_scale / weight_scale_2).
                stem = hf_name.removesuffix(".weight")
                key = _param_key_for(hf_name)
                if key is not None:
                    tensor = dequant_nvfp4(
                        tensor, tensors[stem + ".weight_scale"], tensors[stem + ".weight_scale_2"]
                    )
            elif (
                hf_name.endswith(".weight")
                and hf_name.removesuffix(".weight") + ".weight_scale" in tensors
            ):
                # Per-tensor FP8 (official NVFP4 GDN/attn linears, standalone
                # FP8): f8 weight * scalar scale, kept native like the block
                # format above — the scalar is expanded to the same
                # [ceil(N/128), ceil(K/128)] wscale layout so one kernel
                # serves both.
                stem = hf_name.removesuffix(".weight")
                key = _param_key_for(hf_name)
                if key is not None:
                    ws = tensors[stem + ".weight_scale"].float().reshape(1)
                    n, k = tensor.shape
                    wscale = ws.expand(((n + 127) // 128), ((k + 127) // 128)).contiguous()
                    fp8_native[key] = (tensor, wscale)
                    tensor = dequant_fp8(tensor, wscale).to(torch.bfloat16)
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
                # Quant siblings consumed with their weight tensor above;
                # input_*_scale is activation quant — inference-engine
                # business, ignored at load.
                continue
            else:
                key = _param_key_for(hf_name)
            if key is None:
                continue  # vision tower, MTP head, optimizer state, ...
            if num_layers is not None and key not in specs:
                continue  # truncated-away layer: not required, not loaded
            if key in params:
                raise RuntimeError(
                    f"`{source_desc}`: tensor for param `{key}` appears in multiple "
                    f"shards (last: {shard.name})"
                )
            if key.endswith(".conv1d") and tensor.ndim == 3:
                tensor = tensor.reshape(tensor.shape[0], -1)  # [C,1,K]/[C,K,1] -> [C,K]
            params[key] = tensor

    # Native FP8 weights: keep the e4m3 weight + per-128-block scale alongside
    # the bf16 master (the sm90 prefill path computes with .w8 directly; the
    # master is recording-only, like the fp4 masters).
    for key, (w8, wscale) in fp8_native.items():
        params[f"{key}.w8"] = w8.contiguous()
        params[f"{key}.wscale"] = wscale.contiguous()

    # Embedding/lm_head tying.
    if cfg.tie_word_embeddings:
        params.pop("lm_head", None)  # model reuses embed_tokens
        params.pop("lm_head.w8", None)
        params.pop("lm_head.wscale", None)
    elif "lm_head" not in params:
        raise RuntimeError(f"`{source_desc}`: untied model is missing lm_head.weight")

    # Completeness + shape check against the canonical schema.
    missing = sorted(set(specs) - set(params))
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

    # Quantize the fp4 linears on load (bf16 master kept for the STE backward).
    # Native-fp8 linears are skipped: their checkpoint format is already the
    # sm90 prefill compute format (re-packing to fp4 would lose the e4m3
    # precision and force the K-loop dequant path).
    if cfg.fp4:
        for key in sorted(fp4_param_keys(cfg)):
            if f"{key}.w8" in params:
                continue
            master = params[key]
            if master.dtype != torch.bfloat16:
                master = master.to(torch.bfloat16)
                params[key] = master
            wq, scale = pack_fp4(master)
            params[f"{key}.wq"] = wq
            params[f"{key}.scale"] = scale

    if fuse_projections:
        _fuse_projections(cfg, params)
    return Model(cfg, params)


# --- Checkpoint saving ------------------------------------------------------

#: Param suffix -> HF suffix (reverse of ``_LAYER_SUFFIXES``).
_HF_SUFFIXES = {v: k for k, v in _LAYER_SUFFIXES.items()}


def _hf_tensor_name(key: str) -> str:
    """Param key -> HF tensor name (reverse of :func:`_param_key_for`)."""
    if key == "embed_tokens":
        return "model.language_model.embed_tokens.weight"
    if key == "final_norm":
        return "model.language_model.norm.weight"
    if key == "lm_head":
        return "lm_head.weight"
    _, layer, suffix = key.split(".", 2)
    return f"model.language_model.layers.{layer}.{_HF_SUFFIXES[suffix]}"


def save_hf(model: Model, path: str | Path) -> None:
    """Save params as HF safetensors + config.json (``load_hf`` reads it back).

    fp4 masters are saved bf16 and re-packed on load when ``cfg.fp4``. The
    optimizer state is not saved. # ponytail: training-state checkpoints are
    day-2 (agent-infer's ``checkpoint.rs``).
    """
    from dataclasses import asdict

    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    tensors = {
        _hf_tensor_name(k): t.detach().cpu().contiguous()
        for k, t in model.params.items()
        if k in param_specs(cfg)
    }
    save_file(tensors, str(path / "model.safetensors"))
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
