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
#: sm70 is a staircase, not a line: the GEMV ladder rounds verify width up to a
#: rung, so at ctx 1024 verify costs w<=2 36.58, w<=4 49.87, w<=8 68.46 ms, one
#: draft forward 5.53 (errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md,
#: wins/2026-09-02-draft-is-two-thirds-of-a-spec-tick.md).
#: H20 constants, and on sm70 they change the trim's answer. Measured sm70 cost is
#: 0.670 + 0.5265*W dense ticks = bias 15.9 ms, row 12.5 ms at ctx 1024 -- 13x and
#: 24x off (wins/2026-09-03-verify-tick-cost-is-a-line-in-width.md). The trim reads
#: the RATIO, and H20's makes a row 0.25% of the bias against sm70's 79%, so these
#: over-admit: a low-acceptance batch keeps 2 drafts where the measured price keeps
#: 0 (__main__ asserts both). Runs on every spec tick via _draft_chains -- capture
#: replays the verify, the trim decides what enters it.
#: ponytail: left in place until the reprice is A/B'd on the pod (task #38) -- a
#: serving-behavior flip does not ship on a derivation.
BIAS_MS = 211.0
ROW_MS = 0.53

#: Verify widths the sm70 M-ladder serves without padding waste. A width
#: between rungs pays the next rung's full price: depth 5 (W=6) costs the same
#: 8-row launch as depth 7 (W=8), which measured 10% SLOWER than depth 3 on the
#: one workload where every draft is accepted. 32 is the top rung, and it is
#: no longer a cliff — X is pre-packed f16 there too, 29-36 us/row against the
#: 122-128 it cost when the flag stopped at 8.
LADDER_WIDTHS = (1, 2, 4, 8, 32)


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

    # The trim must REFUSE a chain that cannot pay for its own rows. With the
    # measured sm70 price (a row is 79% of the bias, not H20's 0.25%) a low-
    # acceptance batch keeps nothing; under BIAS_MS/ROW_MS it keeps 2, which is
    # how the mispricing turns into speculating at a loss. Constants here are the
    # measured ones, so this fails if spec.py is repriced without re-measuring.
    sm70 = dict(bias_ms=0.670 * 23.7, row_ms=0.5265 * 23.7)
    assert verify_lens([survival([0.5, 0.25, 0.1])] * 4, **sm70) == [0] * 4
    assert verify_lens([survival([0.95, 0.90, 0.85])] * 4, **sm70) == [2] * 4
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
    skipped: list[str] = []
    with safe_open(str(path), "pt", device="cpu") as f:
        names = list(f.keys())
        # The two formats differ in their RMSNorm convention, and the source name
        # is what tells them apart: a DSpark head carries hidden_norm, a Qwen
        # NextN head carries pre_fc_norm_hidden. See the fold below.
        dspark = any(n.endswith("hidden_norm.weight") for n in names)
        for name in names:
            bare = name.removeprefix("mtp.").removeprefix("model.")
            stem = bare.removesuffix(".weight").removesuffix(".bias")
            if stem in _DRAFT_TOP:
                key = _DRAFT_TOP[stem]
                if key == "confidence":  # the only head with a bias
                    key += ".bias" if bare.endswith(".bias") else ".weight"
                params[key] = f.get_tensor(name)
                continue
            mapped = _param_key_for(bare)
            # forward reads the embedding and the readout off the TRUNK
            # (:132, :148), so a head shipping its own would be dead weight —
            # and engine.py's _quantize_draft packs anything 2D, which at
            # 248320x5120 is 2.5 GB each on a card that has OOMed at 31.3.
            if mapped in ("embed_tokens", "lm_head", "final_norm"):
                skipped.append(bare)
                continue
            if mapped is not None:
                params[mapped] = f.get_tensor(name)
    if skipped:
        warnings.warn(
            f"draft head {path}: ignoring {sorted(skipped)} — the trunk's are shared",
            stacklevel=2,
        )
    # Zero-centered Qwen3_5RMSNorm: load_hf folds the +1 in, this path must too
    # (without it the head's argmax ranked 248191/248320). A DSpark head's norms
    # are plain w*x (dspark.rs:580,726) — folding there corrupts every scale
    # silently, with none of the anti-correlation that made this bug findable.
    if not dspark:
        for k, v in params.items():
            if k.endswith(("norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding")):
                params[k] = (v.float() + 1.0).to(v.dtype)
    missing = {"fc", "norm", "pre_fc_norm_hidden"} - set(params)
    if missing:
        raise RuntimeError(f"draft head {path}: missing {sorted(missing)}")
    # Indices must be 0..n-1: an absolute-index convention (DeepSeek numbers its
    # MTP layer by its position in the trunk) would otherwise infer a depth of
    # index+1 and fail later on a missing layers.0, pointing at the wrong thing.
    idx = sorted({int(k.split(".")[1]) for k in params if k.startswith("layers.")})
    if idx and idx != list(range(len(idx))):
        raise RuntimeError(f"draft head {path}: layers indexed {idx}, expected 0..{len(idx) - 1}")
    return DraftHead(trunk, params, num_layers=len(idx) or 1)
