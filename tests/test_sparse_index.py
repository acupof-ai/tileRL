"""Gates for the V4.1 CSA2 page indexer math (sparse-KV unit D, CPU f32 twin).

- page indexer-K projection groups attention heads into index heads
- page score = sum_h ReLU(q_h.k_h/sqrt(di)), heads merged in the op
- selector-facing scores are [rows, L_src, pages], window pages masked -inf
- 16 full layers -> 4 source groups of 4, selection reused by the group
- KL target pools dense mass per page excluding the window; gradcheck on q/k
"""

from __future__ import annotations

import torch

from tilerl.sparse_index import (
    WINDOW_PAGES,
    index_source_groups,
    indexer_kl,
    indexer_warmup_bwd,
    indexer_warmup_loss,
    page_index_scores,
    page_mass_target,
    page_scores_for_selector,
    project_page_keys,
)

R, L_SRC = 2, 4
IH, DI = 4, 32
DH = 16
H_ATT, D_ATT = 8, 16
PAGES = 12


def _proj():
    torch.manual_seed(0)
    return torch.randn(IH, D_ATT, DI)


def _qk(n_pages: int, q: int = 3):
    torch.manual_seed(1)
    k_pages = torch.randn(R, L_SRC, n_pages, H_ATT, D_ATT)
    iq = torch.randn(R, L_SRC, q, IH, DI)
    return iq, k_pages


def test_source_groups_split_full_layers_into_reused_groups():
    sources, groups = index_source_groups(16, 4)
    assert sources == [0, 4, 8, 12]
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    # every layer covered once; sources are first of their group
    flat = [l for g in groups for l in g]
    assert flat == list(range(16))
    import pytest
    with pytest.raises(ValueError):
        index_source_groups(16, 3)


def test_project_page_keys_groups_heads_and_projects():
    _, k_pages = _qk(PAGES)
    w = _proj()
    ik = project_page_keys(k_pages, w)
    assert ik.shape == (R, L_SRC, PAGES, IH, DI)
    # explicit reference: mean each attention-head group of the page K, then project
    grouped = k_pages.reshape(R, L_SRC, PAGES, IH, H_ATT // IH, D_ATT).mean(4)
    expect = torch.einsum("rlphd,hde->rlphe", grouped, w)
    assert torch.allclose(ik, expect, atol=1e-5)
    # a non-divisible head count is a config error, not a silent truncation
    import pytest
    with pytest.raises(ValueError):
        project_page_keys(torch.randn(R, L_SRC, PAGES, 7, D_ATT), w)


def test_page_scores_are_relu_weighted_sum_over_index_heads():
    iq, k_pages = _qk(PAGES)
    w = _proj()
    ik = project_page_keys(k_pages, w)
    scores = page_index_scores(iq, ik)
    assert scores.shape == (R, L_SRC, 3, PAGES)
    dots = torch.einsum("rlqhd,rlphd->rlqhp", iq, ik) * (DI ** -0.5)
    assert torch.allclose(scores, torch.relu(dots).sum(3), atol=1e-6)
    assert (scores >= 0).all()  # ReLU: a page score is never negative


def test_selector_scores_3d_with_window_masked():
    iq, k_pages = _qk(PAGES)
    w = _proj()
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    ik = project_page_keys(k_pages, w)
    scores = page_scores_for_selector(iq, ik, n_pages)
    assert scores.shape == (R, L_SRC, PAGES)  # exact select_pages input
    # last n_win_pages pages (the window) are -inf, never selected by the indexer
    assert torch.isneginf(scores[..., -WINDOW_PAGES:]).all()
    assert torch.isfinite(scores[..., :-WINDOW_PAGES]).all()
    # rows with fewer valid pages mask the tail beyond their n_pages
    partial = page_scores_for_selector(iq, ik, torch.tensor([6, PAGES]))
    assert torch.isneginf(partial[0, 0, 6:]).all()
    assert torch.isfinite(partial[1, 0, :-WINDOW_PAGES]).all()


def test_page_mass_target_pools_per_page_and_drops_window():
    block = 16
    tokens = PAGES * block
    torch.manual_seed(3)
    # dense per-query attention mass (already summed over attention heads)
    mass = torch.softmax(torch.randn(R, L_SRC, 3, tokens), -1)
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    pooled = page_mass_target(mass, n_pages, block)
    assert pooled.shape == (R, L_SRC, 3, PAGES)
    # L1-normalised over the INDEXABLE pages only
    assert torch.allclose(pooled.sum(-1), torch.ones(R, L_SRC, 3), atol=1e-5)
    # window pages hold zero target mass
    assert (pooled[..., -WINDOW_PAGES:] == 0).all()
    # indexable mass equals the dense tokens inside those blocks, renormalised
    kept = mass.reshape(R, L_SRC, 3, PAGES, block).sum(-1)
    kept[..., -WINDOW_PAGES:] = 0
    assert torch.allclose(pooled, kept / kept.sum(-1, keepdim=True), atol=1e-6)


def test_indexer_kl_gradcheck_on_queries_and_projected_keys():
    torch.manual_seed(4)
    iq = torch.randn(R, L_SRC, 2, IH, DI, dtype=torch.float64, requires_grad=True)
    ik = torch.randn(R, L_SRC, PAGES, IH, DI, dtype=torch.float64, requires_grad=True)
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    target = page_mass_target(
        torch.softmax(torch.randn(R, L_SRC, 2, PAGES * 16, dtype=torch.float64), -1),
        n_pages, 16, WINDOW_PAGES)
    assert torch.autograd.gradcheck(lambda q, k: indexer_kl(q, k, target, n_pages),
                                    (iq, ik), eps=1e-6, atol=1e-4)


def test_warmup_drives_kl_down_on_a_fixed_batch():
    """The warm-up premise: dense page mass is a stationary target and indexer-
    only gradients lower KL. The learnable indexer-Q fits a fixed, positive key
    bank whose page 4 is the teacher (positive keys keep the ReLU gate open as
    it is at convergence). 30 Adam steps must take KL below half (number stated)."""
    torch.manual_seed(5)
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    # positive page keys; the teacher is an INDEXABLE page (window is the last
    # WINDOW_PAGES, so with 12 pages only 0..3 are indexed)
    teacher = 2
    ik = torch.rand(R, L_SRC, PAGES, IH, DI).abs() * 0.1
    ik[:, :, teacher] += 1.0
    ik = ik.detach()
    target = torch.zeros(R, L_SRC, 3, PAGES)
    target[:, :, :, teacher] = 1.0
    iq = (0.1 * torch.randn(R, L_SRC, 3, IH, DI)).abs().requires_grad_()
    opt = torch.optim.Adam([iq], lr=0.1)

    def kl_now():
        return indexer_kl(iq, ik, target, n_pages).item()

    k0 = kl_now()
    for _ in range(30):
        opt.zero_grad()
        indexer_kl(iq, ik, target, n_pages).backward()
        opt.step()
    k1 = kl_now()
    assert k1 < k0 * 0.5, f"KL did not fall by half on the fixed batch: {k0:.4f} -> {k1:.4f}"


# ---------------- part 2: warm-up tape op and its hand-written reverse ----------------

def _warmup_inputs(dtype=torch.float32, seed=7, q=3):
    torch.manual_seed(seed)
    h = torch.randn(R, L_SRC, q, DH, dtype=dtype)
    k_pages = torch.randn(R, L_SRC, PAGES, H_ATT, D_ATT, dtype=dtype)
    iq_w = torch.randn(IH, DH, DI, dtype=dtype)
    ik_w = torch.randn(IH, D_ATT, DI, dtype=dtype)
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    mass = torch.softmax(torch.randn(R, L_SRC, q, PAGES * 16, dtype=dtype), -1)
    target = page_mass_target(mass, n_pages, 16, WINDOW_PAGES)
    return h, k_pages, iq_w, ik_w, target, n_pages


def test_warmup_bwd_gradcheck_on_both_projection_weights():
    """The tape reverse is hand-written, so its two weight gradients are checked
    against f64 central differences. H and page-K are frozen and intentionally
    receive no gradient."""
    h, k_pages, iq_w, ik_w, target, n_pages = (
        t.to(torch.float64) if torch.is_tensor(t) else t
        for t in _warmup_inputs(torch.float64))
    iq_w.requires_grad_()
    ik_w.requires_grad_()
    kw = dict(n_pages=n_pages, n_win_pages=WINDOW_PAGES)

    def loss(iw, kw_):
        return indexer_warmup_loss(h, k_pages, iw, kw_, target, **kw)

    d_iq, d_ik = indexer_warmup_bwd(
        torch.ones((), dtype=torch.float64), iq_w, ik_w, h, k_pages, target, **kw)
    assert torch.autograd.gradcheck(lambda a, b: loss(a, b), (iq_w, ik_w),
                                    eps=1e-6, atol=1e-4)
    # independent finite-difference check of the hand-written reverse itself;
    # perturb raw values, no autograd graph
    iq_w.requires_grad_(False)
    ik_w.requires_grad_(False)
    for w, ana in ((iq_w, d_iq), (ik_w, d_ik)):
        num = torch.zeros_like(w)
        idxs = torch.randperm(w.numel())[:24]
        flat = w.reshape(-1)
        for j in idxs:
            old = flat[j].item()
            flat[j] = old + 1e-6
            fp = indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, **kw).item()
            flat[j] = old - 1e-6
            fm = indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, **kw).item()
            flat[j] = old
            num.reshape(-1)[j] = (fp - fm) / 2e-6
        a = ana.reshape(-1)[idxs]
        n_ = num.reshape(-1)[idxs]
        rel = ((a - n_).norm() / a.norm()).item()
        assert rel < 1e-5, f"warmup bwd rel err {rel:.3e} for {tuple(w.shape)}"


def test_warmup_records_one_tape_entry_and_only_the_two_weights_are_leaves():
    from tilerl import autograd

    h, k_pages, iq_w, ik_w, target, n_pages = _warmup_inputs()
    with autograd.Tape() as tape:
        l = indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, n_pages)
    assert len(tape._entries) == 1
    assert tape._entries[0].op_name == "indexer_warmup"
    assert l.shape == ()  # scalar mean loss seeds the reverse with ones
    grads = tape.backward(torch.ones(()))
    assert set(grads) == {id(iq_w), id(ik_w)}
    assert grads[id(iq_w)].shape == iq_w.shape
    assert grads[id(ik_w)].shape == ik_w.shape
    assert torch.isfinite(grads[id(iq_w)]).all()
    assert torch.isfinite(grads[id(ik_w)]).all()


def test_warmup_tape_step_lowers_the_loss():
    """One optimizer step driven by the tape reverse must lower the frozen-batch
    loss (the recipe's unit of progress); only the two weights change."""
    from tilerl import autograd

    h, k_pages, iq_w, ik_w, target, n_pages = _warmup_inputs()
    params = [iq_w, ik_w]
    opt = autograd.AdamW(lr=0.05)

    def loss_now():
        return indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, n_pages).item()

    l0 = loss_now()
    with autograd.Tape() as tape:
        indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, n_pages)
    grads = tape.backward(torch.ones(()), needs={id(p) for p in params})
    opt.step(params, grads)
    l1 = loss_now()
    assert l1 < l0, f"one warm-up step raised the loss: {l0:.4f} -> {l1:.4f}"


def test_indexer_warmup_recipe_runs_one_step_on_tiny(tmp_path, monkeypatch, capsys):
    """Design acceptance (design-sparse-kv.md): the warm-up recipe runs one step
    on tiny and the manifest gate passes. The frozen base -> capture -> page pool
    -> tape backward -> AdamW chain is exercised through the real CLI."""
    import json

    from tilerl import cli
    from tilerl.ledger import gates_pass

    monkeypatch.setenv("TILERL_RUNS", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["tilerl", "train", "--recipe", "indexer-warmup", "--steps", "1", "--json"])
    cli.main()
    out = capsys.readouterr().out
    m = json.loads(out[out.index("{"):])
    assert m["inputs"]["algo"] == "indexer-warmup"
    assert gates_pass(m), m["gates"]
    gate = m["gates"][0]
    assert gate["name"] == "indexer_warmup_step_runs" and gate["passed"] is True
    import math
    assert math.isfinite(m["metrics"]["kl_first"])
