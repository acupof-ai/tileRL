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
#: wins/2026-09-02-draft-is-two-thirds-of-a-spec-tick.md). Those components rebuild
#: the end-to-end tick to 3-7% at W=2/4/8, so depth moves TWO terms: one more draft
#: forward (5.53 ms, flat) plus a wider verify (4.65-6.64 ms/W, falling as rungs
#: absorb it) -- do not price it as one.
#: H20 constants, and repricing them for sm70 is NOT the fix -- the cost's SHAPE is
#: wrong here, not its scale. engine.py pads every chain to max(len) and the ladder
#: rounds B*W up, so a trim between two widths sharing a rung saves nothing (W=3 and
#: W=4 collide at B=1 and at B=4 alike); only W<=2 is a cheaper rung. The measured
#: price would cut W=4 at acceptance p~=0.92, just below the recorded 84.4%, exactly
#: where the end-to-end numbers say W=4 earns 1.157-1.228x
#: (errors/2026-09-03-repricing-verify-lens-was-the-wrong-fix.md).
#: ponytail: H20 line on a staircase cost. The sm70 line 0.670 + 0.5265*W is fitted
#: at B=1 (bench_ctx_decode.py submits one request), so its W is a chain width and
#: not a launched-row count -- re-measure the slope at B=4 before pricing a trim with
#: it, or the rung and the line are on different axes.
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
    is monotone decreasing, so one global cut yields a prefix per request.

    Prices a tick as ``bias_ms + row_ms * rows`` -- the H20 shape. sm70 instead pays
    a staircase in the WIDEST chain, so the two disagree about the optimal cut; see
    the module constants.
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
    assert verify_lens([[0.99, 0.98, 0.97]], bias_ms=1.0, row_ms=0.1) == [3]
    assert verify_lens([[1e-9, 1e-9]]) == [0]
    lens = verify_lens([[0.99, 0.9, 0.2], [0.3, 0.05, 0.01]], bias_ms=1.0, row_ms=0.1)
    assert lens[0] >= lens[1], lens

    # A trim only pays on sm70 if it changes the RUNG, not the width: engine.py pads to
    # max(len) and B*W rounds up, so a trim between two widths sharing a rung buys
    # nothing. True at B=1 (W=3 and W=4 both launch 4 rows) and at B=4 (both 32), which
    # is what makes repricing the constants the wrong fix. Fails if LADDER_WIDTHS changes.
    for B, collide, cheap in ((1, 4, 2), (4, 32, 8)):
        rung = {w: next(x for x in LADDER_WIDTHS if x >= B * w) for w in (1, 2, 3, 4)}
        assert rung[3] == rung[4] == collide, f"B={B}: W=3 and W=4 must share a rung: {rung}"
        assert rung[2] == cheap, f"B={B}: W=2 must be a cheaper rung: {rung}"

    # The profiled components must still rebuild the end-to-end tick, or the two-term
    # story above is stale. verify(W) + (W-1) draft forwards, against the measured line
    # 0.670 + 0.5265*W dense ticks of 23.7 ms (ctx 1024). Held to 3-7% when written.
    for w, verify_ms in ((2, 36.58), (4, 49.87), (8, 68.46)):
        parts = (verify_ms + (w - 1) * 5.53) / 23.7
        line = 0.670 + 0.5265 * w
        assert abs(parts / line - 1) < 0.10, f"W={w}: parts {parts:.3f} vs line {line:.3f}"
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

    def forward(self, hidden, ids, positions, kv, backend, hidden_out=None,
                last_only=False) -> torch.Tensor:
        """hidden [B,T,H] (trunk's pre-final-norm state), ids [B,T] (the token
        each position predicts FROM) -> draft logits [B,T,vocab], or [B,1,vocab]
        when ``last_only`` selects one position per row. ``hidden_out`` receives
        the head's own hidden at FULL width, appended before the reduction."""
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
        # Same trade the trunk makes (model.py:371): a vocab-wide readout over every
        # prefill position is thrown away one line later. Here it OOMed a 32 GB card --
        # 1.41 GiB at B=8 ctx=512, of which 8 rows (7.6 MiB) were read.
        if last_only is not False and x.shape[1] > 1:
            idx = (torch.full((x.shape[0],), x.shape[1] - 1, device=backend.device)
                   if last_only is True
                   else torch.as_tensor([n - 1 for n in last_only], device=backend.device))
            x = x[torch.arange(x.shape[0], device=backend.device), idx].unsqueeze(1)
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
