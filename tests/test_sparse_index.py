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
    project_indexer_queries,
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


# ---------------- science metric: top-k recall of dense page mass ----------------

def test_topk_recall_is_one_when_k_covers_every_indexable_page():
    from tilerl.sparse_index import topk_page_recall
    iq, k_pages = _qk(PAGES)
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    ik = project_page_keys(k_pages, _proj())
    scores = page_scores_for_selector(iq, ik, n_pages)
    mass = torch.softmax(torch.randn(R, L_SRC, 3, PAGES * 16), -1)
    target = page_mass_target(mass, n_pages)
    # only 12-8 = 4 indexable pages here, so k=4 is dense selection
    rec = topk_page_recall(scores, target, n_pages, k_pages=PAGES - WINDOW_PAGES)
    assert torch.allclose(rec, torch.ones(()), atol=1e-6), rec


def test_topk_recall_counts_mass_on_picked_pages_and_never_the_window():
    from tilerl.sparse_index import topk_page_recall
    # one source layer, one query; put a known score order on 4 indexable pages
    n_pages = torch.full((R,), PAGES, dtype=torch.long)
    scores = torch.full((R, 1, PAGES), float("-inf"))
    # indexable pages 0..3: page 2 hottest, then 0, then 3, then 1
    order = {2: 4.0, 0: 3.0, 3: 2.0, 1: 1.0}
    for pg, v in order.items():
        scores[:, :, pg] = v
    target = torch.zeros(R, 1, 1, PAGES)
    # teacher mass 0.6 on page 2, 0.4 on page 1
    target[:, :, :, 2] = 0.6
    target[:, :, :, 1] = 0.4
    # k=1 picks only page 2 -> recall 0.6; k=3 picks {2,0,3} -> still 0.6;
    # k=4 adds page 1 -> recall 1.0
    r1 = topk_page_recall(scores, target, n_pages, k_pages=1).item()
    r3 = topk_page_recall(scores, target, n_pages, k_pages=3).item()
    r4 = topk_page_recall(scores, target, n_pages, k_pages=4).item()
    assert abs(r1 - 0.6) < 1e-6 and abs(r3 - 0.6) < 1e-6 and abs(r4 - 1.0) < 1e-6
    # window pages carry no teacher mass and are masked out of the pick itself,
    # so even a corrupt selector that emits a finite, hotter window score cannot
    # steal a slot: recall stays 0.6 (page 2), not 0.
    corrupt = scores.clone()
    corrupt[:, :, -1] = 100.0  # window page hotter than everything
    rc = topk_page_recall(corrupt, target, n_pages, k_pages=1).item()
    assert abs(rc - 0.6) < 1e-6, rc


def test_topk_recall_partial_row_masks_its_tail():
    from tilerl.sparse_index import topk_page_recall
    iq, k_pages = _qk(PAGES)
    n_pages = torch.tensor([PAGES - 2, PAGES])  # row 0 has 2 fewer pages
    ik = project_page_keys(k_pages, _proj())
    scores = page_scores_for_selector(iq, ik, n_pages)
    mass = torch.softmax(torch.randn(R, L_SRC, 3, PAGES * 16), -1)
    target = page_mass_target(mass, n_pages)
    # finite recall in [0,1] for a ragged page count, no NaN from the tail
    rec = topk_page_recall(scores, target, n_pages, k_pages=2)
    assert 0.0 <= rec.item() <= 1.0 and torch.isfinite(rec)


def test_streaming_long_sequence_teacher_matches_naive_dense_pooling():
    """The O(T*block)-memory teacher for 8k-32k sequences must equal the naive
    dense [t,t] mass pooled per key page, including a non-block-divisible
    trailing partial page. Checked on a small GQA case."""
    from tilerl.train import _dense_causal_mass, dense_causal_page_mass

    torch.manual_seed(11)
    b, t, hq, hkv, d, block = 1, 49, 4, 2, 8, 16   # 3 full pages + 1-key tail
    q = torch.randn(b, t, hq, d)
    k = torch.randn(b, t, hkv, d)

    naive = _dense_causal_mass(q, k)                       # [b,t,t]
    npages = (t + block - 1) // block                      # 4, not 3
    # reference pooling over ALL t keys, last page padded with zeros
    ref = torch.zeros(b, t, npages)
    full = (t // block) * block
    ref[:, :, : t // block] = naive[:, :, :full].reshape(b, t, t // block, block).sum(-1)
    ref[:, :, t // block] = naive[:, :, full:].sum(-1)
    got = dense_causal_page_mass(q, k, block)              # [b,t,npages]
    assert got.shape == (b, t, npages)
    # With the trailing page included the only gap is f32 accumulation order:
    # maxdiff ~1e-7, so the tolerance is set off that closed gap (not 1e-2).
    assert torch.isfinite(got).all()
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    l1 = got.sum(-1)
    late = l1[0, block:]
    assert torch.allclose(late, torch.ones_like(late), atol=1e-4)
    # a divisible-T case still matches exactly and has no trailing empty page
    got48 = dense_causal_page_mass(q[:, :48], k[:, :48], block)
    assert got48.shape == (b, 48, 3)


def test_indexer_recall_wires_to_the_trained_weights_on_tiny():
    """The runtime recall wrapper must read the SAME iq/ik weights the warm-up
    step updates: training on a frozen batch must not leave recall unchanged in
    the improving direction (guards a detached/rebound weight, the id()-class
    failure the tape docs warn about). Random tiny teacher is near-uniform, so
    this asserts a strict rise, not the 27B 0.9 level."""
    from tilerl_kernels.backend import get_backend

    from tilerl import config, model, train

    be = get_backend()
    m = model.build_random(config.tiny(), seed=0)
    gen = torch.Generator().manual_seed(0)
    hkv, dk, hid, di = m.cfg.num_kv_heads, m.cfg.head_dim, m.cfg.hidden_size, 16
    w = {"iq": 0.1 * torch.randn(hkv, hid, di, generator=gen),
         "ik": 0.1 * torch.randn(hkv, dk, di, generator=gen)}
    ids = torch.randint(0, m.cfg.vocab_size, (1, 256), generator=gen)
    before = train.indexer_recall(m, ids, be, w, k_pages_pick=2)["index"]
    opt = train.AdamW(lr=0.02)
    for _ in range(20):
        train.indexer_warmup_step(m, ids, be, w, opt)
    rec = train.indexer_recall(m, ids, be, w, k_pages_pick=2)
    after = rec["index"]
    assert set(rec) == {"index", "bounds"} and 0.0 <= rec["bounds"] <= 1.0
    assert after > before, f"recall did not improve with training: {before:.3f} -> {after:.3f}"
    assert 0.0 <= before <= 1.0 and 0.0 <= after <= 1.0


def test_warmup_capture_handles_a_non_block_divisible_sequence():
    """A trailing partial key page must not be dropped (52's #512 change) nor
    break the query/key axis alignment: the full warm-up path must run for T not
    a multiple of the block, with ceil pages and teacher rows over the real T."""
    from tilerl_kernels.backend import get_backend

    from tilerl import config, model, train

    be = get_backend()
    m = model.build_random(config.tiny(), seed=0)
    gen = torch.Generator().manual_seed(0)
    hkv, dk, hid, di = m.cfg.num_kv_heads, m.cfg.head_dim, m.cfg.hidden_size, 16
    w = {"iq": 0.1 * torch.randn(hkv, hid, di, generator=gen),
         "ik": 0.1 * torch.randn(hkv, dk, di, generator=gen)}
    ids = torch.randint(0, m.cfg.vocab_size, (1, 249), generator=gen)  # 249 = 15*16+9
    H, k_pages, target, n_pages, q_eval, bounds = train.indexer_capture(
        m, ids, be, 16, WINDOW_PAGES)
    assert int(n_pages[0]) == 16              # ceil(249/16)
    assert H.shape[2] == 249                  # queries unpadded
    assert target.shape[2] == 249
    assert k_pages.shape == (1, 1, 16, hkv, dk)
    assert bounds.shape == (1, 1, 16, hkv, 2, dk) and q_eval.shape[2] == 249
    assert torch.isfinite(target).all() and torch.isfinite(k_pages).all()
    assert torch.isfinite(bounds).all()
    # recall and one training step run on the ragged tensors without an axis error
    assert 0.0 <= train.indexer_recall(m, ids, be, w, 2)["index"] <= 1.0
    loss = train.indexer_warmup_step(m, ids, be, w, train.AdamW(lr=0.02))
    assert loss == loss  # not NaN


def test_indexer_held_recall_from_a_prepared_dir(tmp_path):
    """The cross-corpus control loader must read held-only spans from a prep dir
    and return a finite recall per length under given weights."""
    import json

    from tilerl_kernels.backend import get_backend

    from tilerl import config, model, train

    be = get_backend()
    m = model.build_random(config.tiny(), seed=0)
    gen = torch.Generator().manual_seed(0)
    w = train.init_indexer_weights(m.cfg, gen, be.device, 16)
    d = tmp_path / "corpus"
    d.mkdir()
    for split, n in (("held", 2), ("train", 3)):  # train spans must be ignored
        with open(d / f"{split}_256.jsonl", "w") as fh:
            for _ in range(n):
                fh.write(json.dumps({"ctx": 256,
                                     "ids": torch.randint(0, m.cfg.vocab_size, (256,)).tolist()}) + "\n")
    rec = train.indexer_held_recall(m, be, d, w, k_pages_pick=2)
    assert set(rec) == {"256"}
    r = rec["256"]
    assert set(r) == {"index", "bounds"}
    for sc in ("index", "bounds"):
        st = r[sc]
        assert set(st) == {"mean", "min", "per_span"} and len(st["per_span"]) == 2
        assert 0.0 <= st["mean"] <= 1.0 and st["mean"] == st["mean"]  # finite


def test_bounds_scorer_recall_is_one_at_full_k_and_self_consistent():
    """The training-free Quest scorer uses the SAME selector/teacher as the
    learned scorer, so with k covering every indexable page its recall must be 1
    (every earlier page selected), matching the engine's full-k==dense
    equivalence; and bounds recall must lie in [0,1] at a partial k."""
    from tilerl_kernels.backend import get_backend

    from tilerl import config, model, train
    from tilerl.sparse_index import WINDOW_PAGES

    be = get_backend()
    m = model.build_random(config.tiny(), seed=0)
    gen = torch.Generator().manual_seed(0)
    w = train.init_indexer_weights(m.cfg, gen, be.device, 16)
    ids = torch.randint(0, m.cfg.vocab_size, (1, 256), generator=gen)
    n_indexable = 256 // 16 - WINDOW_PAGES      # 16 - 8 = 8
    full = train.indexer_recall(m, ids, be, w, k_pages_pick=n_indexable)
    assert abs(full["bounds"] - 1.0) < 1e-6, full
    part = train.indexer_recall(m, ids, be, w, k_pages_pick=2)
    assert 0.0 <= part["bounds"] <= 1.0


def test_indexer_projections_take_bf16_activations_with_f32_weights():
    """GPU feeds bf16 frozen-base H/K but the indexer head is f32 (the smoke
    failed 'expected BFloat16 but got Float' before the boundary cast). Both
    projections and the full warm-up loss must accept the dtype mix on a
    bf16-capable device (CPU torch supports bf16 compute)."""
    torch.manual_seed(0)
    r, l, pages, ih, m, da, dh, di, win = 1, 1, 6, 2, 2, 8, 16, 8, 1
    hk = ih * m
    h = torch.randn(r, l, 4, dh, dtype=torch.bfloat16)
    k_pages = torch.randn(r, l, pages, hk, da, dtype=torch.bfloat16)
    iq_w = 0.1 * torch.randn(ih, dh, di)
    ik_w = 0.1 * torch.randn(ih, da, di)
    iq = project_indexer_queries(h, iq_w)
    ik = project_page_keys(k_pages, ik_w)
    assert iq.dtype == torch.float32 and ik.dtype == torch.float32
    n_pages = torch.full((r,), pages, dtype=torch.long)
    mass = torch.softmax(torch.randn(r, l, 4, pages * 16), -1)
    target = page_mass_target(mass, n_pages, n_win_pages=win)
    loss = indexer_warmup_loss(h, k_pages, iq_w, ik_w, target, n_pages,
                               n_win_pages=win)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    diq, dik = indexer_warmup_bwd(None, iq_w, ik_w, h, k_pages, target, n_pages,
                                 n_win_pages=win)
    assert diq.dtype == torch.float32 and dik.dtype == torch.float32
    assert torch.isfinite(diq).all() and torch.isfinite(dik).all()


def test_sampled_query_teacher_matches_full_rows():
    """The 27B amendment: dense_causal_page_mass at q_positions must equal the
    full teacher's rows at exactly those positions (subsample is O(nq*T), not a
    different teacher). sample_query_positions is seeded, distinct and >= min."""
    from tilerl.train import dense_causal_page_mass, sample_query_positions

    torch.manual_seed(12)
    b, t, hq, hkv, d, block = 1, 97, 4, 2, 8, 16
    q = torch.randn(b, t, hq, d)
    k = torch.randn(b, t, hkv, d)
    full = dense_causal_page_mass(q, k, block)
    qp = sample_query_positions(t, 23, seed=7, min_pos=40)
    assert qp.shape == (23,) and qp.unique().numel() == 23
    assert int(qp.min()) >= 40 and bool((qp[1:] >= qp[:-1]).all())
    sub = dense_causal_page_mass(q, k, block, qp)
    assert torch.allclose(sub, full.index_select(1, qp), atol=1e-6)
    # the same seed gives the same positions; a different seed differs
    assert torch.equal(qp, sample_query_positions(t, 23, 7, 40))
    assert not torch.equal(qp, sample_query_positions(t, 23, 8, 40))
    # asking for more positions than the pool yields every eligible position
    allpos = sample_query_positions(t, 10000, 7, 40)
    assert int(allpos[0]) == 40 and int(allpos[-1]) == t - 1 and allpos.numel() == t - 40
