"""Gates for the learned KV indexer math (sparse-KV unit D, CPU f32 twin).

- the page-score contract matches the bounds scorer: [rows, L, q, pages] f32
- max-pool: a page wins iff one of its tokens is hot; short last page padded
- target is dense attention mass summed over heads and L1-normalised
- gradcheck: the KL's gradients on index query/key are real (numerical vs tape)
- warm-up shape: on a fixed batch, fitting only the indexer drives KL down
"""

from __future__ import annotations

import torch

from tilerl.sparse_index import (
    BLOCK_TOKENS,
    dense_mass_target,
    index_page_scores,
    index_token_logits,
    indexer_kl,
    page_maxpool,
)

R, L, QH, D = 2, 3, 4, 16


def _iq_ikey(k: int, *, requires_grad: bool = True, q: int = 5):
    torch.manual_seed(0)
    iq = torch.randn(R, L, q, QH, D, requires_grad=requires_grad)
    ikey = torch.randn(R, L, k, D, requires_grad=requires_grad)
    return iq, ikey


def test_page_scores_contract_rows_layers_query_pages():
    iq, ikey = _iq_ikey(k=2 * BLOCK_TOKENS + 3)
    scores = index_page_scores(iq, ikey)
    assert scores.shape == (R, L, 5, 3)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()


def test_page_maxpool_hot_token_selects_its_page():
    # One token at page 1 dominates; every page's score is the max over its block.
    toks = torch.full((2, 2 * BLOCK_TOKENS), -10.0)
    toks[0, BLOCK_TOKENS + 2] = 5.0
    pages = page_maxpool(toks, BLOCK_TOKENS)
    assert pages.shape == (2, 2)
    assert pages[0, 0].item() == -10.0
    assert pages[0, 1].item() == 5.0
    # A short context pads to one page with -inf tail (kept finite by the real token).
    short = page_maxpool(torch.tensor([[-10.0, -10.0]]), BLOCK_TOKENS)
    assert short.shape == (1, 1) and torch.isfinite(short).all()
    assert torch.isneginf(page_maxpool(torch.empty(1, 0), BLOCK_TOKENS)).all()


def test_token_logits_sum_heads_and_scale():
    iq, ikey = _iq_ikey(k=7)
    out = index_token_logits(iq, ikey)
    # explicit reference: einsum expansion needs [R,L,q,H,1,D] x [R,L,1,1,D,k]
    dots = iq.unsqueeze(4) @ ikey.transpose(-1, -2).reshape(R, L, 1, 1, D, -1)
    expect = dots.squeeze(4).sum(3) * (D ** -0.5)
    assert out.shape == (R, L, 5, 7)
    assert torch.allclose(out, expect, atol=1e-5)
    # one head, one query token and one key token: self logit is scaled D
    iq1 = torch.ones(1, 1, 1, 1, D)   # [R,L,q,H,D]
    ik1 = torch.ones(1, 1, 1, D)      # [R,L,k,D]
    assert torch.allclose(index_token_logits(iq1, ik1), torch.tensor([[[[D * D ** -0.5]]]]))


def test_dense_target_sums_heads_and_normalises():
    torch.manual_seed(1)
    logits = torch.randn(R, L, QH, 5, 11)  # heads, q, k
    t = dense_mass_target(logits)
    assert t.shape == (R, L, 5, 11)
    assert torch.allclose(t.sum(-1), torch.ones(R, L, 5), atol=1e-5)
    # equal to the mean per-head softmax (sum/QH == mean), not softmax of summed logits
    mean_softmax = torch.softmax(logits, -1).mean(2)
    assert not torch.allclose(t, torch.softmax(logits.sum(2), -1))
    assert torch.allclose(t, mean_softmax / mean_softmax.sum(-1, keepdim=True).clamp_min(1e-12))


def test_indexer_kl_gradcheck_on_queries_and_keys():
    # f64 central-differences oracle over aligned query/key inputs.
    iq64 = torch.randn(R, L, 5, QH, D, dtype=torch.float64, requires_grad=True)
    ik64 = torch.randn(R, L, 9, D, dtype=torch.float64, requires_grad=True)
    target = dense_mass_target(torch.randn(R, L, QH, 5, 9, dtype=torch.float64)).detach()
    assert torch.autograd.gradcheck(lambda a, b: indexer_kl(a, b, target),
                                    (iq64, ik64), eps=1e-6, atol=1e-4)


def test_warmup_drives_kl_down_on_a_fixed_batch():
    """The warm-up premise: dense mass is a stationary target and indexer-only
    updates lower KL. The learnable piece is the index query projection applied
    to fixed queries, scored against fixed keys whose per-query target is one
    sharp page — the learnable optimum is a softmax concentrated there. 20 SGD
    steps on one fixed batch; KL must fall by half (number stated, no band)."""
    torch.manual_seed(2)
    rows, layers, q, k = R, L, 5, 16
    queries = torch.randn(rows, layers, q, QH, D)
    ikey = torch.randn(rows, layers, k, D)
    # A teacher whose mass concentrates on token index 4 for every query: the
    # index projection's job is to make that token's summed-head dot largest.
    teacher = torch.zeros(rows, layers, q, k)
    teacher[..., 4] = 8.0
    target = torch.softmax(teacher, -1)

    proj = (0.1 * torch.randn(D, D)).requires_grad_()
    opt = torch.optim.SGD([proj], lr=1.0)

    def kl_now():
        return indexer_kl(queries @ proj, ikey, target).item()

    k0 = kl_now()
    for _ in range(20):
        opt.zero_grad()
        indexer_kl(queries @ proj, ikey, target).backward()
        opt.step()
    k1 = kl_now()
    assert k1 < k0 * 0.5, f"KL did not fall by half on the fixed batch: {k0:.4f} -> {k1:.4f}"
