"""DeepSeek-V4.1 CSA2 learned page indexer (sparse-KV unit D, CPU f32 twin).

The unit of indexing is the PAGE, not the token (ckl ruling 2026-09-11):

- an **indexer-K** is projected per page from that page's K — one key per
  16-token block per source layer (V4.1 projects from an m-token entry);
- an **indexer-Q** is projected from the layer input H with ``ih`` heads of
  dim ``di`` (a distinct projection from the attention Q);
- a page's score is ``sum_h ReLU(q_h . k_h) / sqrt(di)``;
- selection is computed at a few INDEX SOURCE layers and reused by the layers
  in each group, so one hot set serves a group and a cold-page fetch is paid
  once;
- the local window (last ``n_win`` tokens, ``n_win_pages`` pages) is attended
  regardless, so its pages are EXCLUDED from the indexer scores and KL target;
- one softmax runs over [selected pages ; window] in sparse attention.

This module is the scorer only: pure f32 torch math (the CPU twin of the
registry op). It emits ``[rows, L_src, pages]`` — exactly ``select_pages``'s
input — and the per-query-per-page attention mass pooled per page for the
warm-up KL. Selection, the window union and the sparse softmax live in the
bounds/selector unit; entry compression, cross-layer KV reuse and the
hierarchical candidate pool are deferred from V4.1.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kv_cache import BLOCK_TOKENS

#: Index query heads and per-head dim (V4.1-Flash releases 32/128; one H20
#: holds 4/128 per the sparse-KV design). Kept as module constants for the
#: derived byte rows; tiny gates use small values.
INDEX_HEADS = 4
INDEX_HEAD_DIM = 128
#: Number of index SOURCE layers over the 16 full-attention layers: one source
#: per group of four, selection reused by the group's other three.
INDEX_SOURCE_LAYERS = 4
#: Local window always attended: 128 tokens = 8 pages.
WINDOW_TOKENS = 128
WINDOW_PAGES = WINDOW_TOKENS // BLOCK_TOKENS


def index_source_groups(n_full_layers: int,
                        n_sources: int = INDEX_SOURCE_LAYERS) -> tuple[list[int], list[list[int]]]:
    """The layer -> source-group map. ``n_sources`` source layers each serve a
    contiguous group of full-attention layers; source s is the first layer of
    group s and selection is reused for the rest. Returns ``(source_layer_ids,
    groups)``.

    16 layers / 4 sources -> sources [0,4,8,12], groups [[0,1,2,3],[4,5,6,7],...].
    """
    if n_full_layers % n_sources:
        raise ValueError(
            f"{n_full_layers} full-attn layers must divide into {n_sources} sources")
    groups: list[list[int]] = [[] for _ in range(n_sources)]
    for layer in range(n_full_layers):
        groups[layer * n_sources // n_full_layers].append(layer)
    source_layers = [g[0] for g in groups]
    return source_layers, groups


def project_page_keys(k_pages: Tensor, ik_weight: Tensor) -> Tensor:
    """Project indexer-K per page from the page's K (V4.1 ik_weight).

    ``k_pages``   [rows, L_src, pages, h_attn, d_attn] — one K summary per page
                    (the mean over the page's tokens; V4.1 uses a learned combine
                    over its m entries, deferred)
    ``ik_weight`` [ih, d_attn, di]

    Groups attention heads into the ``ih`` index heads (mean of the group), then
    projects: returns [rows, L_src, pages, ih, di].
    """
    r, l, p, h, d = k_pages.shape
    ih = ik_weight.shape[0]
    if h % ih:
        raise ValueError(f"{h} attention heads do not divide into {ih} index heads")
    grouped = k_pages.reshape(r, l, p, ih, h // ih, d).mean(dim=4)  # mean of group
    # weight is per index head [ih, d_attn, di]; contract only that head's dims
    return torch.einsum("rlphd,hde->rlphe", grouped, ik_weight)


def page_index_scores(iq: Tensor, ik: Tensor) -> Tensor:
    """The V4.1 page score: sum_h ReLU(q_h . k_h) / sqrt(di).

    ``iq`` [rows, L_src, q, ih, di] — indexer-Q projected from the layer input
    ``ik`` [rows, L_src, pages, ih, di] — projected page indexer-K

    Returns [rows, L_src, q, pages]. The ReLU is the indexer's gating product;
    heads are merged inside the op (the selector sees no head axis). The
    selector-facing form max-pools the query axis in
    :func:`page_scores_for_selector`.
    """
    di = iq.shape[-1]
    dots = torch.einsum("rlqhd,rlphd->rlqhp", iq, ik) * (di ** -0.5)
    return torch.relu(dots).sum(dim=3)


def page_scores_for_selector(iq: Tensor, ik: Tensor,
                             n_pages: Tensor, n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """The exact ``select_pages`` input: [rows, L_src, pages] f32.

    Max over query positions (a page hot for any query is selectable), with the
    local window's last ``n_win_pages`` and every page >= ``n_pages`` masked to
    -inf — the window is always attended separately, so it is never indexed.
    """
    r, l, q, ih, di = iq.shape
    pages = ik.shape[2]
    scores = page_index_scores(iq, ik).amax(dim=2)  # max over queries
    page_id = torch.arange(pages, device=iq.device)
    valid = page_id[None, None, :] < n_pages.to(iq.device)[:, None, None]
    not_window = page_id[None, None, :] < (n_pages.to(iq.device)[:, None, None]
                                           - n_win_pages)
    scores = scores.masked_fill(~(valid & not_window), float("-inf"))
    return scores


def page_mass_target(attn_mass: Tensor, n_pages: Tensor,
                     block_tokens: int = BLOCK_TOKENS,
                     n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """The warm-up KL target per source layer: dense attention mass pooled per
    PAGE, excluding the always-attended window.

    ``attn_mass`` [rows, L_src, q, tokens] — dense softmax mass summed over
    attention heads, L1-normalised per query. Returns
    [rows, L_src, q, pages]: the mass of each indexable page (token masses
    summed within the block; window pages zeroed), re-normalised over the
    indexable pages so KL is over the selection decision, not the window.
    """
    r, l, q, t = attn_mass.shape
    pad = (-t) % block_tokens
    if pad:
        attn_mass = torch.cat([attn_mass, torch.zeros(r, l, q, pad,
                                                      device=attn_mass.device)], dim=-1)
    pages = attn_mass.shape[-1] // block_tokens
    pooled = attn_mass.reshape(r, l, q, pages, block_tokens).sum(dim=-1)
    valid = torch.arange(pages, device=attn_mass.device)[None, None, None, :] \
        < n_pages.to(attn_mass.device)[:, None, None, None]
    indexable = valid & (torch.arange(pages, device=attn_mass.device)[None, None, None, :]
                         < (n_pages[:, None, None, None] - n_win_pages))
    pooled = pooled * indexable
    return pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def indexer_kl(iq: Tensor, ik: Tensor, target_page_mass: Tensor,
               n_pages: Tensor, n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """Mean KL(dense page mass || softmax(indexer page logits)) over queries.

    Logits and target are both masked to the indexable (non-window) pages, so
    the softmax the indexer learns is exactly the selection distribution; the
    target is detached by the caller. Gradients flow into ``iq``/``ik``.
    """
    pages = ik.shape[2]
    page_id = torch.arange(pages, device=iq.device)
    upper = n_pages.to(iq.device) - n_win_pages  # first indexable page boundary
    indexable = (page_id[None, :] < n_pages.to(iq.device)[:, None]) & \
        (page_id[None, :] < upper[:, None])    # [rows, pages]
    indexable4 = indexable[:, None, None, :]   # broadcast [rows,1,q,pages]
    # The SELECTOR masks with true -inf (a masked page is never picked), but the
    # KL softmax uses a large finite sentinel: log_softmax(-inf) has no finite
    # numerical derivative, so central-differences gradcheck returns NaN, and the
    # target puts zero mass there anyway, making ~-1e9 and -inf equivalent.
    logits = page_index_scores(iq, ik).masked_fill(~indexable4, -1e9)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target_page_mass * logp).sum(dim=-1).mean()
