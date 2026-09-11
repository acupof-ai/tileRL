"""Learned KV indexer for sparse attention (DeepSeek V3.2 lightning indexer).

The indexer scores context pages for selection. Per full-attention layer it
holds one shared fp8 index key per token (128 B, shared by the index query
heads); the query side has a few heads. Token logits are the summed-head dot
of index query against the shared key; the selector max-pools token logits
over each page (``kv_cache.BLOCK_TOKENS``) and takes the top pages.

Shapes use the contract shared with the bounds scorer: page scores are
``[rows, L, q, pages]`` f32 (``L`` = full-attention layers only), so a single
selector consumes either scorer. This module is f32 torch math — the CPU twin
of the registry op; the sm90 indexer is a later unit. Nothing here touches the
paged pool yet.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kv_cache import BLOCK_TOKENS

#: The shared per-token index key width (bytes on device: fp8 + one f32 scale).
INDEX_KEY_DIM = 128
#: Number of index query heads per layer (V3.2 uses a small head count).
INDEX_QUERY_HEADS = 4


def index_token_logits(iq: Tensor, ikey: Tensor) -> Tensor:
    """Per-token index logits: summed-head dot of index queries vs the shared key.

    ``iq``   [rows, L, q, H, D] — the query-side index heads (one row per query token)
    ``ikey`` [rows, L, k, D]    — the one shared key per context token

    Returns [rows, L, q, k] f32, scaled by 1/sqrt(D) like an attention score,
    summed over heads so the selector and the KL target see one mass per token.
    """
    d = iq.shape[-1]
    # No dtype cast: the CPU twin is called with f32 in production and f64 in
    # gradcheck; a .float() here would silently defeat numerical gradcheck.
    # [rows,L,q,H,k] -> sum H -> [rows,L,q,k]
    return torch.einsum("rlqhd,rlkd->rlqk", iq, ikey) * (d ** -0.5)


def page_maxpool(token_scores: Tensor, block_tokens: int = BLOCK_TOKENS) -> Tensor:
    """Max-pool per-token scores into per-page scores.

    ``token_scores`` [..., k] -> [..., pages], one page per ``block_tokens``
    context tokens; a short final page is padded with -inf so it cannot win
    unless it is the only content. Max, not mean: a page is selectable if ANY
    token in it is hot, which is the attention bound the selector approximates.
    """
    *lead, k = token_scores.shape
    pad = (-k) % block_tokens
    if pad:
        token_scores = torch.cat(
            [token_scores, token_scores.new_full((*lead, pad), float("-inf"))], dim=-1)
    pages = token_scores.shape[-1] // block_tokens
    return token_scores.reshape(*lead, pages, block_tokens).amax(dim=-1)


def index_page_scores(iq: Tensor, ikey: Tensor,
                      block_tokens: int = BLOCK_TOKENS) -> Tensor:
    """The scorer contract: token logits -> per-page ``[rows, L, q, pages]`` f32."""
    return page_maxpool(index_token_logits(iq, ikey), block_tokens)


def dense_mass_target(attn_scores: Tensor) -> Tensor:
    """The warm-up KL target: full attention's softmax mass, summed over heads,
    L1-normalised per query.

    ``attn_scores`` [rows, L, H, q, k] (pre-softmax logits). Returns
    [rows, L, q, k] — a per-token probability distribution the indexer softmax
    is fit to. Summing per-head softmax vectors (not logits) is the dense mass
    the page recall acceptance measures against.
    """
    mass = torch.softmax(attn_scores.float(), dim=-1).sum(dim=2)  # sum heads
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def indexer_kl(iq: Tensor, ikey: Tensor, target_mass: Tensor) -> Tensor:
    """Mean KL(target_mass || softmax(index logits)) over queries.

    Target is detached by the caller (it comes from the frozen dense forward);
    gradients flow only into ``iq``/``ikey``. The cross-entropy of the target
    distribution under the indexer logits is KL up to target entropy.
    """
    logits = index_token_logits(iq, ikey)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target_mass * logp).sum(dim=-1).mean()
