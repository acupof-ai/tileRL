"""Speculative decoding: the draft head and the verify-length policy.

``verify_lens`` decides how many drafted tokens per request are worth verifying
this tick (DSpark §3.2.2, sglang's ``compute_verify_token_budget``): a draft
costs a trunk row whether or not it is accepted, so maximize goodput
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms. ``survival[j]`` = P(the first j+1 drafts all accept).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

#: One trunk verify forward = fixed + per-row cost, ms. agent-infer's H20 numbers.
BIAS_MS = 211.0
ROW_MS = 0.53


def survival(confidences: list[float]) -> list[float]:
    out, p = [], 1.0
    for c in confidences:
        p *= float(c)
        out.append(p)
    return out


def verify_lens(
    survivals: list[list[float]], bias_ms: float = BIAS_MS, row_ms: float = ROW_MS
) -> list[int]:
    """Per-request draft-keep lengths maximizing verify goodput. ``survivals[r]``
    is monotone decreasing, so one global cut yields a prefix per request."""
    eps = 1e-6
    r = len(survivals)
    flat = sorted((p for s in survivals for p in s if p >= eps), reverse=True)
    best, cut, total = r / (bias_ms + row_ms * r), float("inf"), 0.0
    for i, p in enumerate(flat, 1):
        total += p
        theta = (r + total) / (bias_ms + row_ms * (r + i))
        if theta > best:
            best, cut = theta, p
    out = []
    for s in survivals:
        n = 0
        while n < len(s) and s[n] >= cut:
            n += 1
        out.append(n)
    return out


if __name__ == "__main__":  # runnable check
    assert survival([0.9, 0.8, 0.5]) == [0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5]
    assert verify_lens([[0.99, 0.98, 0.97]], bias_ms=1.0, row_ms=0.1) == [3]
    assert verify_lens([[1e-9, 1e-9]]) == [0]
    lens = verify_lens([[0.99, 0.9, 0.2], [0.3, 0.05, 0.01]], bias_ms=1.0, row_ms=0.1)
    assert lens[0] >= lens[1], lens
    print("spec: verify_lens OK", lens)


class DraftHead:
    """NextN / DSpark draft head: ``fc([norm(embed(t)), norm(h_trunk)])`` into a
    short full-attention stack, read out through the trunk's lm_head. The layers
    are a ``Model`` with a 1-layer config, not a second transformer block."""

    def __init__(self, trunk: Any, params: dict[str, torch.Tensor], num_layers: int = 1) -> None:
        from .model import Model

        self.trunk = trunk
        self.params = params
        cfg = replace(
            trunk.cfg, num_layers=num_layers, full_attn_layers=tuple(range(num_layers)), fp4=False
        )
        self.cfg = cfg
        self.layers = Model(cfg, params)
        self.has_confidence = "confidence.weight" in params

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None) -> torch.Tensor:
        """hidden [B,T,H] (trunk's pre-final-norm state), ids [B,T] (the token
        each position predicts FROM) -> draft logits [B,T,vocab]. ``hidden_out``
        receives the head's own hidden, which the next draft position consumes."""
        eps = self.cfg.rms_eps
        ids = torch.as_tensor(ids, dtype=torch.long, device=backend.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        e = backend.embedding(ids, self.trunk.params["embed_tokens"])
        if "pre_fc_norm_embedding" in self.params:  # Qwen NextN: both sides normed
            e = backend.rmsnorm(e, self.params["pre_fc_norm_embedding"], eps)
        hidden = backend.rmsnorm(hidden, self.params["pre_fc_norm_hidden"], eps)
        # embed first (agent-infer qwen35_spec.rs:40-55); the other order does not predict
        x = self.layers._linear(backend, torch.cat([e, hidden], dim=-1), "fc")
        for i in range(self.cfg.num_layers):
            x = self.layers._full_attn(i, x, positions, kv, backend)
            x = self.layers._mlp(i, x, kv, backend)
        if hidden_out is not None:
            hidden_out.append(x)
        x = backend.rmsnorm(x, self.params["norm"], eps)
        head = "embed_tokens" if self.trunk.cfg.tie_word_embeddings else "lm_head"
        return self.trunk._linear(backend, x, head)

    def confidence(self, hidden, probs, backend) -> torch.Tensor:
        """Per-position P(accept), [B,T]: the checkpoint's head, else ``probs``."""
        if not self.has_confidence:
            return probs
        y = backend.linear(hidden, self.params["confidence.weight"],
                           bias=self.params.get("confidence.bias"))
        return torch.sigmoid(y).reshape(y.shape[:-1])


#: Draft tensor stems -> param keys, matched after any ``layers.N.`` prefix.
#: Qwen NextN prefixes ``mtp.``; DSpark drops it and carries one ``hidden_norm``
#: instead of the two pre-fc norms.
_DRAFT_TOP = {
    "fc": "fc",
    "norm": "norm",
    "hidden_norm": "pre_fc_norm_hidden",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
    "confidence_head.proj": "confidence",
}


def _split_layer(stem: str) -> tuple[str, str]:
    """``layers.3.mlp_conv.base_kernel`` -> ``("layers.3.", "mlp_conv.base_kernel")``."""
    if stem.startswith("layers."):
        idx, sep, tail = stem[len("layers.") :].partition(".")
        if sep and idx.isdigit():
            return f"layers.{int(idx)}.", tail
    return "", stem


def read_head_params(path: str | Path, stems: dict[str, str]) -> dict[str, torch.Tensor]:
    """One draft-head safetensors -> param keys: ``stems`` names the head's own
    tensors, ``_param_key_for`` the ordinary Qwen3 layer ones."""
    from safetensors import safe_open

    from .model import _param_key_for

    params: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unknown: list[str] = []
    nextn = False
    with safe_open(str(path), "pt", device="cpu") as f:
        for name in list(f.keys()):
            bare = name.removeprefix("mtp.").removeprefix("model.")
            stem = bare.removesuffix(".weight").removesuffix(".bias")
            nextn |= stem == "pre_fc_norm_hidden"
            prefix, tail = _split_layer(stem)
            key = stems.get(tail)
            if key is not None:
                if key == "confidence":  # the only head tensor with a bias
                    key += ".bias" if bare.endswith(".bias") else ".weight"
                params[prefix + key] = f.get_tensor(name)
                continue
            mapped = _param_key_for(bare)
            # forward reads the embedding and the readout off the TRUNK, so a head
            # shipping its own is dead weight — and engine._quantize_draft packs
            # anything 2-D, which at 248320x5120 is 2.5 GB on a card that has OOMed.
            if mapped in ("embed_tokens", "lm_head", "final_norm"):
                skipped.append(bare)
            elif mapped is not None:
                params[mapped] = f.get_tensor(name)
            else:
                unknown.append(bare)
    if skipped:
        warnings.warn(
            f"draft head {path}: ignoring {sorted(skipped)} — the trunk's are shared",
            stacklevel=2,
        )
    # A tensor this map does not name is the wrong reader for this checkpoint, not
    # dead weight: loading a DFlash2 head through _DRAFT_TOP drops all 11 of its
    # conv and selector weights, and the first draft then dies on a KeyError far
    # from the cause.
    if unknown:
        raise RuntimeError(
            f"draft head {path}: {len(unknown)} tensor(s) map to no parameter — "
            f"{sorted(unknown)[:8]}{'...' if len(unknown) > 8 else ''}. Wrong head "
            "format for this reader, or a key this port does not implement."
        )
    # Zero-centered Qwen3_5RMSNorm (y = x*(1+w)): load_hf folds the +1 in for the
    # trunk, and only a Qwen NextN head is built that way. DSpark and DFlash norms
    # are plain w*x — agent-infer's dspark.rs:580,726, and vLLM/sglang build every
    # DFlash norm from their stock RMSNorm. Keying the fold on the one format that
    # needs it makes no-fold the default, which is the safe way round: the missing
    # fold is loud (the head's argmax ranked 248191/248320), the spurious one is not.
    if nextn:
        for k, v in params.items():
            if k.endswith(("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")):
                params[k] = (v.float() + 1.0).to(v.dtype)
    return params


def load_draft(trunk: Any, path: str | Path) -> Any:
    """Load a draft head from one safetensors file beside the trunk: a Qwen
    NextN / DSpark chain head, or the DFlash2 block drafter."""
    from safetensors import safe_open

    with safe_open(str(path), "pt", device="cpu") as f:
        if any(n.startswith("candidate_selector.") for n in list(f.keys())):
            from .dflash2 import load_dflash2

            return load_dflash2(trunk, path)
    params = read_head_params(path, _DRAFT_TOP)
    missing = {"fc", "norm", "pre_fc_norm_hidden"} - set(params)
    if missing:
        raise RuntimeError(f"draft head {path}: missing {sorted(missing)}")
    # Indices must be 0..n-1: an absolute-index convention (DeepSeek numbers its MTP
    # layer by its position in the trunk) would otherwise infer a depth of index+1
    # and fail later on a missing layers.0, pointing at the wrong thing.
    idx = sorted({int(k.split(".")[1]) for k in params if k.startswith("layers.")})
    if idx and idx != list(range(len(idx))):
        raise RuntimeError(f"draft head {path}: layers indexed {idx}, expected 0..{len(idx) - 1}")
    return DraftHead(trunk, params, num_layers=len(idx) or 1)
