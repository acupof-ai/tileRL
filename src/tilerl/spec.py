"""Speculative decoding: the draft head and the verify-length policy.

Two pieces, separable:

``verify_lens`` decides HOW MANY drafted tokens per request are worth verifying
this tick — DSpark §3.2.2 (sglang's ``compute_verify_token_budget``). Verifying
a draft costs a row in the trunk forward whether or not it is accepted, so the
question is goodput, not acceptance rate: maximize
``(R + Σ top-B survival) / (bias + row·(R + B))`` over the admission cut. B=0
is one of the arms, so the policy never chooses to speculate when speculating
loses.

``survival[j]`` is P(the first j+1 drafts all accept) — monotone decreasing, the
cumulative product of per-position confidence. A checkpoint with a
``confidence_head`` supplies that confidence directly; a DFlash-style head has
none, and the draft's own softmax probability for the token it emitted is the
fallback.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

__all__ = ["verify_lens", "survival", "DraftHead", "load_draft"]

#: Measured cost of one trunk verify forward: a fixed cost plus a per-row cost,
#: in ms. The defaults are agent-infer's H20 numbers; re-measure per target.
#: V100 sm70 is NOT well described by this two-term form — see
#: errors/2026-09-01-spec-depth-is-a-staircase-not-a-line.md: the sm70 GEMV
#: ladder rounds the verify width up to 1/2/4/8, so cost is a staircase in
#: depth and a linear model mis-prices every width that is not a rung.
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
    """Per-position confidence -> P(first j+1 drafts all accept)."""
    out, p = [], 1.0
    for c in confidences:
        p *= float(c)
        out.append(p)
    return out


def verify_lens(
    survivals: list[list[float]], bias_ms: float = BIAS_MS, row_ms: float = ROW_MS
) -> list[int]:
    """Per-request draft-keep lengths maximizing verify goodput.

    ``survivals[r]`` must be monotone decreasing (it is a cumulative product),
    which is what makes a single global admission cut yield a PREFIX per
    request rather than an arbitrary subset.
    """
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
    # A confident draft is worth verifying; a hopeless one is not.
    assert verify_lens([[0.99, 0.98, 0.97]], bias_ms=1.0, row_ms=0.1) == [3]
    assert verify_lens([[1e-9, 1e-9]]) == [0]
    # Cheap rows -> keep more; the cut is global, the kept span is a prefix.
    lens = verify_lens([[0.99, 0.9, 0.2], [0.3, 0.05, 0.01]], bias_ms=1.0, row_ms=0.1)
    assert lens[0] >= lens[1], lens
    print("spec: verify_lens OK", lens)


# --- draft head ---------------------------------------------------------------


class DraftHead:
    """A NextN / DSpark draft head: ``fc([norm(embed(t)), norm(h_trunk)])`` into
    a short stack of full-attention layers, read out through the TRUNK's
    lm_head (the draft never carries its own vocabulary projection).

    Structurally a full-attn layer of the trunk, so the layers reuse
    ``Model._full_attn`` / ``Model._mlp`` verbatim — the head is a Model with a
    1-layer config, not a second implementation of a transformer block.

    Optional heads, probed rather than assumed: ``confidence_head.proj`` gives
    per-position confidence (DSpark); a DFlash-style checkpoint has none and
    the draft's own softmax probability stands in.
    """

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
        each position predicts FROM). Returns draft logits [B,T,vocab];
        ``hidden_out`` receives the head's own hidden, which is what the NEXT
        draft position consumes."""
        eps = self.cfg.rms_eps
        # Model.forward does this for its own inputs; the head is entered
        # directly, so it converts its own.
        ids = torch.as_tensor(ids, dtype=torch.long, device=backend.device)
        positions = torch.as_tensor(positions, dtype=torch.long, device=backend.device)
        e = backend.embedding(ids, self.trunk.params["embed_tokens"])
        if "pre_fc_norm_embedding" in self.params:  # Qwen NextN: both sides normed
            e = backend.rmsnorm(e, self.params["pre_fc_norm_embedding"], eps)
        hidden = backend.rmsnorm(hidden, self.params["pre_fc_norm_hidden"], eps)
        # fc consumes concat(norm_embed(t), norm_hidden(h)) — embed first, per
        # agent-infer's qwen35_spec.rs:40-55; the other order looks like a head
        # that simply does not predict.
        # Through the Model seam, not backend.linear: fc is served in whatever
        # format the head was quantized to, exactly like every other projection.
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
        """Per-position P(this draft is accepted), [B,T].

        The checkpoint's own head when it has one; otherwise ``probs``, the
        draft's own probability for the token it emitted (spec.py docstring)."""
        if not self.has_confidence:
            return probs
        y = backend.linear(hidden, self.params["confidence.weight"],
                           bias=self.params.get("confidence.bias"))
        return torch.sigmoid(y).reshape(y.shape[:-1])


#: Draft tensor names -> our param keys. The Qwen NextN head prefixes
#: everything with ``mtp.``; DSpark checkpoints drop the prefix and carry one
#: ``hidden_norm`` instead of the two pre-fc norms.
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
    # A Qwen NextN head's norms are zero-centered Qwen3_5RMSNorm (y = x*(1+w)):
    # load_hf folds the +1 in at load, reading the head's file directly skips it,
    # and a head whose norms scale by ~0.2 instead of ~1.2 emits logits
    # ANTI-correlated with the trunk (its argmax ranked 248191/248320).
    # A DSpark head's are PLAIN w*x — agent-infer loads its hidden_norm/norm/
    # input_layernorm/post_attention_layernorm with load_vec_any (dspark.rs:580,
    # 726) and only q_norm/k_norm with load_vec_minus_one. Folding there
    # corrupts every scale silently, with none of the anti-correlation that made
    # the first version of this bug findable.
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
