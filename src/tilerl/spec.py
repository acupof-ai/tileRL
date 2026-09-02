"""Speculative decoding: the draft head and the verify-length policy.

``verify_lens`` decides how many drafted tokens per request are worth verifying
this tick (DSpark §3.2.2, sglang's ``compute_verify_token_budget``): a draft
costs a trunk row whether or not it is accepted, so maximize goodput
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms. ``survival[j]`` = P(the first j+1 drafts all accept).
"""

from __future__ import annotations

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


#: Draft tensor names -> param keys. Qwen NextN prefixes ``mtp.``; DSpark
#: drops it and carries one ``hidden_norm`` instead of the two pre-fc norms.
_DRAFT_TOP = {
    "fc": "fc",
    "norm": "norm",
    "hidden_norm": "pre_fc_norm_hidden",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden",
    "pre_fc_norm_embedding": "pre_fc_norm_embedding",
    "confidence_head.proj": "confidence",
}


def load_draft(trunk: Any, path: str | Path) -> DraftHead:
    """Load a draft head from one safetensors file beside the trunk."""
    from safetensors import safe_open

    from .model import _param_key_for

    params: dict[str, torch.Tensor] = {}
    with safe_open(str(path), "pt", device="cpu") as f:
        for name in list(f.keys()):
            bare = name.removeprefix("mtp.").removeprefix("model.")
            stem = bare.removesuffix(".weight").removesuffix(".bias")
            if stem in _DRAFT_TOP:
                key = _DRAFT_TOP[stem]
                if key == "confidence":  # the only head with a bias
                    key += ".bias" if bare.endswith(".bias") else ".weight"
                params[key] = f.get_tensor(name)
                continue
            mapped = _param_key_for(bare)
            if mapped is not None:
                params[mapped] = f.get_tensor(name)
    # Zero-centered Qwen3_5RMSNorm: load_hf folds the +1 in, this path must too
    # (without it the head's argmax ranked 248191/248320).
    for k, v in params.items():
        if k.endswith(("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")):
            params[k] = (v.float() + 1.0).to(v.dtype)
    missing = {"fc", "norm", "pre_fc_norm_hidden"} - set(params)
    if missing:
        raise RuntimeError(f"draft head {path}: missing {sorted(missing)}")
    n = 1 + max((int(k.split(".")[1]) for k in params if k.startswith("layers.")), default=0)
    return DraftHead(trunk, params, num_layers=n)
