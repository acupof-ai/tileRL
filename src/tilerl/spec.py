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

import numpy as np
import torch

from .kv_cache import BLOCK_TOKENS, BatchKv

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

    #: Drafter contract, shared with ``dflash2.DFlash2Head`` and read by the engine.
    #: ``aux_layers`` are trunk layers whose output the head taps ( () = none, so the
    #: head serves behind a prefix cache); ``width`` is the verify tick's width,
    #: 1 committed token + width-1 drafts; ``no_quant`` stays out of the fp8 serve.
    aux_layers: tuple[int, ...] = ()
    no_quant: tuple[str, ...] = ()

    def set_depth(self, depth: int | None) -> None:
        """Apply the caller's ``spec_depth``; None keeps the head's own. Idempotent."""
        if depth is not None:
            self.width = depth + 1

    def attach(self, backend, num_blocks: int) -> None:
        """The draft KV plane spans the trunk's whole block space, so the head attends
        over the same prefix the trunk does (a chain-local block dropped acceptance
        from 84.4% to 55.8%)."""
        from .kv_cache import PagedKvPool

        self.backend = backend
        self.kv = PagedKvPool(num_blocks, self.cfg.num_kv_heads, self.cfg.head_dim,
                              num_layers=self.cfg.num_layers, device=backend.device,
                              layer_map=tuple(range(self.cfg.num_layers)))

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
        self.width = 3  # 2 drafts; ``set_depth`` overrides

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
        head = self.trunk.cfg.head_key
        return self.trunk._linear(backend, x, head)

    def confidence(self, hidden, probs, backend) -> torch.Tensor:
        """Per-position P(accept), [B,T]: the checkpoint's head, else ``probs``."""
        if not self.has_confidence:
            return probs
        y = backend.linear(hidden, self.params["confidence.weight"],
                           bias=self.params.get("confidence.bias"))
        return torch.sigmoid(y).reshape(y.shape[:-1])

    def step(self, rows) -> None:
        """Contract: leave next tick's chain in ``r.drafts``.

        Draft over every position a row materialized but has not drafted yet:
        position q consumes the trunk hidden at q-1 and the token at q, so the
        run spans ``[draft_pos+1 .. seq_len-1]`` and its last position drafts
        the next token. Leaves next tick's chain in ``r.drafts``."""
        backend = self.backend
        dev = backend.device
        plan = []
        for r in rows:
            if r.hidden is None or r.done:
                continue
            lo, hi = max(1, r.draft_pos + 1), r.seq_len - 1
            if hi < lo:
                continue
            plan.append((r, lo, hi))
        if not plan:
            return
        w = max(hi - lo + 1 for _, lo, hi in plan)
        nb = max(len(r.blocks) for r, _, _ in plan)
        n = len(plan)
        ids = np.zeros((n, w), dtype=np.int64)
        pos = np.zeros((n, w), dtype=np.int64)
        bt = torch.zeros(n, nb, dtype=torch.long)
        hs, sl, sq = [], [], []
        for i, (r, lo, hi) in enumerate(plan):
            q = hi - lo + 1
            ids[i, :q] = r.tokens[lo : hi + 1]
            pos[i, :q] = np.arange(lo, hi + 1)
            bt[i, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.long)
            sl.append(hi + 1)
            sq.append(q)
            # hidden at [lo-1 .. hi-1]; hidden_prev supplies the previous forward's position
            h, base = r.hidden, r.hidden_from
            if r.hidden_prev is not None:
                h, base = torch.cat([r.hidden_prev, r.hidden], dim=1), base - 1
            off = (lo - 1) - base
            hs.append(torch.nn.functional.pad(h[:, off : off + q], (0, 0, 0, w - q)))
        kv = BatchKv(
            block_table=bt.to(dev), seq_len=torch.tensor(sl, device=dev),
            state_slot=torch.zeros(n, dtype=torch.long, device=dev),
            kv_pool=self.kv, state_pool=None,
            seq_q_lens=torch.tensor(sq, device=dev),
        )
        dh: list = []
        logits = self.forward(torch.cat(hs, dim=0), ids, pos, kv, backend,
                                     hidden_out=dh)
        last = torch.tensor([q - 1 for q in sq], device=dev)
        rng = torch.arange(n, device=dev)
        tok, prob = backend.greedy(logits[rng, last].unsqueeze(1))
        h = dh[-1][rng, last].unsqueeze(1)
        confs: list[list[float]] = [[] for _ in plan]
        if (self.width - 1) > 1:
            conf = self.confidence(h, prob, backend)
            for i, c in enumerate(conf[:, -1].tolist()):
                confs[i].append(float(c))
        chains = [[int(t)] for t in tok[:, -1].tolist()]
        for i, (r, _, hi) in enumerate(plan):
            if r.draft_pos == 0:
                # Position 0 is never drafted but attention still reads its page,
                # which a recycled block leaves holding another request's.
                b = r.blocks[0]
                self.kv.k_pool[:, b, :, 0, :] = 0
                self.kv.v_pool[:, b, :, 0, :] = 0
            r.draft_pos = hi

        # Remaining chain steps, one position each, bounded by the blocks the row owns.
        # ponytail: clamps the chain instead of allocating; a row at a block boundary drafts shorter.
        for j in range(1, (self.width - 1)):
            live = [i for i, (r, _, hi) in enumerate(plan)
                    if hi + j < len(plan[i][0].blocks) * BLOCK_TOKENS]
            if not live:
                break
            li = torch.tensor(live, device=dev)
            kv = BatchKv(
                block_table=bt[live].to(dev),
                seq_len=torch.tensor([plan[i][2] + 1 + j for i in live], device=dev),
                state_slot=torch.zeros(len(live), dtype=torch.long, device=dev),
                kv_pool=self.kv, state_pool=None,
                seq_q_lens=torch.ones(len(live), dtype=torch.long, device=dev),
            )
            dh = []
            logits = self.forward(
                h[li], np.array([[chains[i][-1]] for i in live], dtype=np.int64),
                np.array([[plan[i][2] + j] for i in live], dtype=np.int64),
                kv, backend, hidden_out=dh,
            )
            tok, prob = backend.greedy(logits)
            conf = self.confidence(dh[-1], prob, backend)
            for k, c in enumerate(conf[:, -1].tolist()):
                confs[live[k]].append(float(c))
            for k, t in enumerate(tok[:, -1].tolist()):
                chains[live[k]].append(int(t))
            h = h.index_copy(0, li, dh[-1])

        keep = verify_lens([survival(c) for c in confs]) if (self.width - 1) > 1 \
            else [1] * len(plan)
        for i, (r, _, _) in enumerate(plan):
            p = r.params
            if p.max_think_tokens is not None and p.end_think_ids and not r.thought_closed:
                keep[i] = 0  # a forced end-think token is not the sampler's
            r.drafts = chains[i][: keep[i]]


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
    NextN / DSpark chain head, or the DFlash2 block drafter. A checkpoint
    directory resolves to its ``model.safetensors``; mmapping the directory
    itself raises a bare ``OSError: No such device``, which names nothing."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"draft head: {path.parent} holds no model.safetensors")
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
