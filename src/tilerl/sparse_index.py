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
#: Default selected hot pages per row for serve/build_engine. The shipped value
#: tracks the output-fidelity-vs-k table (27B recall at 32k: bounds 0.158 / oracle
#: 0.296 at k=128); bump the one constant when the table picks a larger k.
DEFAULT_SPARSE_K = 128


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
    # The indexer is an f32 head over bf16 frozen-base activations (GPU): cast at
    # the projection boundary. f32->f32 (CPU tiny) is a no-op.
    k_pages = k_pages.to(ik_weight.dtype)
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


def topk_page_recall(selector_scores: Tensor, target_page_mass: Tensor,
                     n_pages: Tensor, k_pages: int,
                     n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """Top-k recall of dense page mass — the warm-up's science metric.

    ``selector_scores`` [rows, L_src, pages] from :func:`page_scores_for_selector`
    (window/invalid already -inf); ``target_page_mass`` [rows, L_src, q, pages]
    from :func:`page_mass_target` (L1-normalised over indexable pages). The
    selector picks at most ``k_pages`` indexable pages per row/layer (shared
    across that layer's queries — a page hot for ANY query is taken); recall is
    the target mass sitting on picked pages, averaged over rows, source layers
    and queries. ``k_pages`` >= the number of indexable pages is dense
    selection and recall is 1 by construction.
    """
    r, l, pages = selector_scores.shape
    indexable = _indexable_mask(n_pages, pages, n_win_pages, selector_scores.device)
    # Pick per row/layer among indexable pages; mask non-indexable below the
    # finite scores so topk never returns the window.
    pick = selector_scores.masked_fill(~indexable[:, None, :], float("-inf")).topk(
        min(k_pages, pages), dim=-1).indices                        # [r,l,k]
    chosen = torch.zeros(r, l, pages, dtype=torch.bool, device=selector_scores.device)
    chosen.scatter_(-1, pick, True)
    chosen &= indexable[:, None, :]                                  # belt and braces
    mass_on = torch.einsum("rlp,rlqp->rlq", chosen.float(), target_page_mass)
    return mass_on.mean()


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
    return exclude_window_renorm(pooled, n_pages, n_win_pages)


def exclude_window_renorm(pooled: Tensor, n_pages: Tensor,
                          n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """Zero window/invalid pages in an already-pooled ``[r,L,q,pages]`` mass and
    L1-normalise over the indexable pages. Shared by the token-pooling target and
    the long-sequence teacher that streams page masses directly."""
    r, l, q, pages = pooled.shape
    indexable = _indexable_mask(n_pages, pages, n_win_pages, pooled.device)
    pooled = pooled * indexable[:, None, None, :]
    return pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _indexable_mask(n_pages: Tensor, pages: int, n_win_pages: int,
                    device) -> Tensor:
    """[rows, pages] bool: a valid page strictly before the last ``n_win_pages``
    (the window). The selector and the KL softmax mask the same set."""
    n = n_pages.to(device)
    page_id = torch.arange(pages, device=device)
    return (page_id[None, :] < n[:, None]) & (page_id[None, :] < n[:, None] - n_win_pages)


def indexer_kl(iq: Tensor, ik: Tensor, target_page_mass: Tensor,
               n_pages: Tensor, n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """Mean KL(dense page mass || softmax(indexer page logits)) over queries.

    Logits and target are both masked to the indexable (non-window) pages, so
    the softmax the indexer learns is exactly the selection distribution; the
    target is detached by the caller. Gradients flow into ``iq``/``ik``.
    """
    pages = ik.shape[2]
    indexable4 = _indexable_mask(n_pages, pages, n_win_pages, iq.device)[:, None, None, :]
    # The SELECTOR masks with true -inf (a masked page is never picked), but the
    # KL softmax uses a large finite sentinel: log_softmax(-inf) has no finite
    # numerical derivative, so central-differences gradcheck returns NaN, and the
    # target puts zero mass there anyway, making ~-1e9 and -inf equivalent.
    logits = page_index_scores(iq, ik).masked_fill(~indexable4, -1e9)
    logp = torch.log_softmax(logits, dim=-1)
    return -(target_page_mass * logp).sum(dim=-1).mean()


def project_indexer_queries(h: Tensor, iq_weight: Tensor) -> Tensor:
    """Project indexer-Q per query from the layer input H (V4.1 indexer_q):
    ``h`` [rows, L_src, q, d_hidden], ``iq_weight`` [ih, d_hidden, di] ->
    [rows, L_src, q, ih, di]. Distinct from the attention-Q projection. Casts
    bf16 frozen activations to the f32 indexer weight dtype at the boundary."""
    return torch.einsum("rlqd,hde->rlqhe", h.to(iq_weight.dtype), iq_weight)


def indexer_warmup_loss(h: Tensor, k_pages: Tensor, iq_weight: Tensor,
                        ik_weight: Tensor, target_page_mass: Tensor,
                        n_pages: Tensor, n_win_pages: int = WINDOW_PAGES) -> Tensor:
    """The warm-up objective as ONE differentiable function of the two indexer
    projection weights (the frozen base's H and page-K carry no gradient):

        iq = H @ iq_weight ; ik = page-K head-grouped @ ik_weight
        loss = KL(dense page mass || softmax(indexer page scores))

    Recorded as the single tape op ``indexer_warmup`` whose reverse is
    :func:`indexer_warmup_bwd`; the warm-up trains these two weights only.
    """
    from .autograd import maybe_record

    iq = project_indexer_queries(h, iq_weight)
    ik = project_page_keys(k_pages, ik_weight)
    loss = indexer_kl(iq, ik, target_page_mass, n_pages, n_win_pages)
    maybe_record("indexer_warmup", loss, iq_weight, ik_weight, h=h, k_pages=k_pages,
                 target_page_mass=target_page_mass, n_pages=n_pages,
                 n_win_pages=n_win_pages)
    return loss


def indexer_warmup_bwd(grad: Tensor, iq_weight: Tensor, ik_weight: Tensor,
                       h: Tensor, k_pages: Tensor, target_page_mass: Tensor,
                       n_pages: Tensor, n_win_pages: int = WINDOW_PAGES
                       ) -> tuple[Tensor, Tensor]:
    """Hand-written reverse of :func:`indexer_warmup_loss` (no torch.autograd):
    softmax-CE delta -> ReLU gate -> the two projection einsums. Returns
    ``(d iq_weight, d ik_weight)``; H and page-K are frozen and get nothing.

    The forward activations are recomputed from the saved inputs, matching the
    tape's recompute-instead-of-store convention. ``grad`` is the scalar upstream
    (ones for a standalone loss); the mean already carries the 1/N factor.
    """
    del grad  # scalar mean: upstream is 1
    r, l, p, ha, da = k_pages.shape
    ih = ik_weight.shape[0]
    m = ha // ih
    if ha % ih:
        raise ValueError(f"{ha} attention heads do not divide into {ih} index heads")
    scale = iq_weight.shape[-1] ** -0.5

    iq = project_indexer_queries(h, iq_weight)              # [r,L,q,ih,di]
    # f32 head over bf16 frozen activations (GPU); same boundary cast as fwd.
    k_pages = k_pages.to(ik_weight.dtype)
    h = h.to(iq_weight.dtype)
    grouped = k_pages.reshape(r, l, p, ih, m, da).mean(4)  # [r,L,p,ih,da]
    ik = torch.einsum("rlphd,hde->rlphe", grouped, ik_weight)
    dots = torch.einsum("rlqhe,rlphe->rlqhp", iq, ik) * scale
    # softmax-CE delta against the SAME -1e9 sentinel and indexable mask as fwd.
    logits = torch.relu(dots).sum(3)
    mask = _indexable_mask(n_pages, p, n_win_pages, h.device)
    probs = torch.softmax(logits.masked_fill(~mask[:, None, None, :], -1e9), dim=-1)
    n = r * l * iq.shape[2]
    # probs is already ~0 at masked pages (exp(-1e9) underflows) and the target
    # is exactly 0 there (page_mass_target), so (probs-target) needs no re-mask.
    delta = (probs - target_page_mass) / n              # [r,L,q,p]
    # score = sum_h relu(dots); gate on the pre-ReLU sign.
    ddots = (delta[:, :, :, None, :] * (dots > 0))         # [r,L,q,ih,p]
    diq = scale * torch.einsum("rlqhp,rlphe->rlqhe", ddots, ik)
    dik = scale * torch.einsum("rlqhp,rlqhe->rlphe", ddots, iq)
    d_iq_weight = torch.einsum("rlqhe,rlqd->hde", diq, h)
    d_ik_weight = torch.einsum("rlphe,rlphd->hde", dik, grouped)
    return d_iq_weight, d_ik_weight

