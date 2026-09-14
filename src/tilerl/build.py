"""The single assembler: config + checkpoint into a Model, and model + backend
into an Engine. Nothing above L4 assembles; cli and the offline apps call here."""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

from . import config as config_mod
from . import memory, precision
from . import model as model_mod
from .engine import (
    Engine,
    StepLimits,
    _graph_on,
    _serve_draft,
    _weight_fingerprint,
    card_guard,
)
from .kv_cache import (
    BLOCK_TOKENS,
    LinearStatePool,
    NoPrefixStore,
    PagedKvPool,
    PrefixStore,
)
from .kv_tiers import DramSnapshots, HostKvPages, KvBootStore
from .sparse_index import DEFAULT_SPARSE_K

QWEN38_SOURCE = os.environ.get("TILERL_QWEN38_SOURCE", "Qwen/Qwen3-27B")

NO_WEIGHTS = (
    "hint: download the checkpoint (or set TILERL_QWEN38_SOURCE to a\n"
    "      local safetensors directory), or use --model tiny."
)

#: Every name build_model builds. The single source the argparse choices import.
MODEL_NAMES = ("tiny", "tiny-agent", "qwen38-27b")


def kv_fp8_dtype(name: str | None):
    """The --kv-fp8 flag value as a torch dtype (None when unset)."""
    if not name:
        return None
    return {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[name]


def _shard(cfg, model, tp: int, backend, model_mod):
    """Every rank builds the WHOLE model and keeps its slice.

    Wasteful and deliberate: sharding at load time needs a loader that reads
    per-rank slices out of the checkpoint, and that is a separate change. On the
    27B this costs each rank a transient full copy.
    # ponytail: whole-model build then slice, per-rank checkpoint reads when the
    # 27B's transient copy is the binding constraint
    """
    if tp <= 1:
        return cfg, model
    from .tensor_parallel import Mesh, shard_params, tp_config

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world % tp:
        raise SystemExit(f"--tp {tp} does not divide WORLD_SIZE={world}")
    mesh = Mesh(dp=world // tp, tp=tp, rank=rank)
    tp_groups, dp_groups = [], []
    for r in range(world):
        m = Mesh(dp=world // tp, tp=tp, rank=r)
        for g, seen in ((m.tp_group(), tp_groups), (m.dp_group(), dp_groups)):
            if g not in seen:
                seen.append(g)
    backend.init_tp(world, rank, tp_groups, dp_groups)
    return tp_config(cfg, tp), model_mod.Model(
        tp_config(cfg, tp), shard_params(model.params, cfg, mesh.tp_rank, tp)
    )


def build_model(
    model_name: str,
    seed: int,
    fuse_projections: bool = False,
    keep_master: bool = False,
    tp: int = 1,
    backend=None,
):
    """(cfg, model): serving fuses projections, training keeps the bf16 masters.

    ``tp`` > 1 shards both here, because a model can only be sharded where it is
    built: the config's head counts must already be divided before any layer
    reshapes with them.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"unknown model {model_name!r}; expected one of {', '.join(MODEL_NAMES)}. "
            f"A local 27B checkpoint is selected with --model qwen38-27b plus "
            f"TILERL_QWEN38_SOURCE=<dir>, not by passing its path here"
        )
    if model_name == "qwen38-27b":
        cfg = config_mod.qwen38_27b()
        try:
            model = model_mod.load_hf(
                cfg, QWEN38_SOURCE, fuse_projections=fuse_projections, keep_master=keep_master
            )
        except Exception as exc:
            print(
                f"error: could not load Qwen3-27B weights from {QWEN38_SOURCE!r}: {exc}\n"
                f"{NO_WEIGHTS}",
                file=sys.stderr,
            )
            sys.exit(1)
        return _shard(cfg, model, tp, backend, model_mod)
    # tiny-agent is tiny with room for one real agent turn; see config.tiny().
    cfg = config_mod.tiny(65536) if model_name == "tiny-agent" else config_mod.tiny()
    model = model_mod.build_random(
        cfg, seed=seed, fuse_projections=fuse_projections, keep_master=keep_master
    )
    return _shard(cfg, model, tp, backend, model_mod)


def fit_blocks(
    cfg, backend, io, cap: int, draft_layers: int = 0, kv_fp8: torch.dtype | None = None
) -> int:
    """KV blocks that fit the free memory left after the weights and the GDN pools.

    Called with the state pool already allocated, so free memory is measured, not
    estimated -- and only after ``empty_cache``, without which the allocator's load
    reserve hides 13 GiB and this returns 64 blocks. The default it replaces was
    ``ctx * max_batch``, which on the 27B's own 262144-token limit is 275 GB of f32
    KV; every bench passed num_blocks explicitly, so serve was the one caller that
    ever saw it.

    ``draft_layers`` is the draft's own layer count, 0 for a dense engine. The draft
    mirrors ``num_blocks`` into a pool of its own (``DraftHead.attach``), and that
    pool's planes are not free: at the 27B's 16 full-attn planes one draft layer is
    1/16 of the trunk's bytes, so charging for it when there is no draft under-fits
    every training engine, and not charging when there is one over-fits serve.

    Holds back a third of what is free: ``PrefixStore`` takes a quarter of what
    remains after this, and the attention partials are transient and scale with B*S.
    ``cap`` wins over the 64-block floor -- a caller asking for fewer means it.
    """
    if backend.device.type != "cuda":
        return cap or 256
    # The single block-size formula and the free*2/3 rule live in tilerl.memory: the fit
    # here and the plan's kv_pool row must not each rederive per-block bytes.
    from .memory import fit_num_blocks

    return fit_num_blocks(
        cfg, torch.cuda.mem_get_info()[0], io, kv_fp8, draft_layers=draft_layers, cap=cap
    )


def build_engine(
    cfg,
    model: Any,
    backend: Any,
    *,
    num_blocks: int = 64,
    num_slots: int = 8,
    max_batch: int = 8,
    max_total_tokens: int = 8192,
    max_num_batched_tokens: int = 512,
    max_blocks: int = 0,
    prefix_store: Any = None,
    #: host-tier budget for demoted GDN snapshots; 0 is off, which stays the default. A
    #: single conversation loses by it: it re-reads only its newest entry, so the LRU
    #: snapshot a demotion picks is never asked for again -- measured, 43 demotions and 0
    #: promotions, with the wall clock 1.51x worse. The condition that reverses it is
    #: `concurrent sessions > HBM snapshot budget` (measured: 2 -> 0 promotions, 9 -> 17,
    #: 12 -> 24). `--dram-bytes` exposes it because that is a property of the deployment,
    #: not of this call site, and an operator who cannot set it has to edit source to run
    #: the multi-session case at all. `/health`'s `dram_promotions` is what says the
    #: workload crossed the threshold; `dram_budget` says the tier is on.
    dram_bytes: int = 0,
    #: pinned-host budget for the sparse-KV cold page tier; 0 is off (dense pool).
    #: When set, PagedKvPool.demote_page/promote_page move whole pages (every plane
    #: of one block id, fp8 scales included) through a HostKvPages tier and the
    #: freed device block goes back to the SAME pool — there is no second pool.
    kv_cold_bytes: int = 0,
    #: cold pages past the pinned host budget spill to this one mmap'd file (serving
    #: spill, block-id keyed; distinct from ssd_path which is the prefix-boot store).
    cold_ssd_path: str = "",
    #: countable SSD spill capacity for admission in bytes. 0 with a spill path
    #: means auto = free space on the spill filesystem; off when no path is set.
    cold_ssd_bytes: int = 0,
    #: dtype for a demoted page's K/V in the host/SSD tier: "native" keeps the pool
    #: dtype, "f16" narrows an f32 pool (sm70, which has no f16 attention path) to
    #: f16 on the D2H copy and widens back on promote. "" picks f16 on an f32 pool
    #: and native everywhere else — sm70's host has half the room and needs it,
    #: sm90's bf16 already is 16-bit. The fp8 scale planes stay f32 either way.
    cold_format: str = "",
    #: directory for the cold-start KV boot store ("" = off). A context saved there
    #: via Engine.save_boot is bulk-reloaded on a later serve whose request prefix
    #: matches exactly, skipping its prefill. Keyed by the same rolling prefix hash,
    #: gated by the weight fingerprint and a per-page checksum.
    kv_store: str = "",
    #: weight/config fingerprint for the --kv-store cold-start boot store. Must
    #: change whenever the weights do — a boot context computed under other weights
    #: is silently wrong, and the fingerprint is the only thing that stops it.
    #: Defaults to the model's shape, which does NOT cover a different checkpoint
    #: at the same shape.
    #: # ponytail: shape-derived fingerprint; hash the weights when two checkpoints of
    #: #   one architecture are served from one boot dir
    ssd_fingerprint: str = "",
    #: HBM budget for resident GDN snapshots; 0 keeps the quarter-of-free rule below.
    state_bytes: int = 0,
    #: fp8 dtype for the KV planes; None is off, the default. 65536 -> 33280 bytes per token
    #: at the 27B's 16x4x256, a 1.969x saving after one f32 scale per
    #: (plane, block, kv_head, token) -- the only grid a single-launch fused writer can
    #: reduce (docs/design-fp8-kv.md). Off because the attention kernels still read a
    #: dequantized plane, so this is capacity, not yet bandwidth.
    kv_fp8: torch.dtype | None = None,
    #: sparse-KV selection, now the DEFAULT. sparse_k>0 selects this many earlier pages
    #: per row + the 8-page window; the device pool holds just that hot set plus chunk
    #: headroom, cold pages demote to a pinned host tier and promote on selection.
    #: scorer "bounds" is the training-free Quest path (Unit F); "index" is a later PR.
    #: Pass sparse_k=0 for the legacy dense engine. A host cold tier is auto-attached
    #: when sparse_k>0 and kv_cold_bytes is unset.
    sparse_k: int = DEFAULT_SPARSE_K,
    scorer: str = "bounds",
    sparse_device_select: bool | None = None,
    decode_graph: bool | None = None,
    draft: Any = None,
    #: Hybrid (sparse_k>0 only): prompts at most this long run DENSE on the
    #: captured graph and pin their whole context (no sparse sharing); longer ones
    #: run sparse. 0 = all requests sparse, the pre-hybrid behavior.
    sparse_min_tokens: int = 0,
    #: Hybrid sparse-prefill chunk cap in tokens; 0 uses the 192 default (~1 s on
    #: the V100 sparse prefill rate). Ignored when sparse_min_tokens is 0.
    sparse_prefill_tokens: int = 0,
    spec_depth: int | None = None,
    decode: Any = None,
) -> Engine:
    """Wire a model + backend into an Engine; pool shapes come from ``cfg``.
    ``decode_graph`` None auto-enables the captured decode tick on CUDA.
    ``num_blocks`` 0 fits the KV pool to free memory, capped at ``max_blocks``."""
    if backend.device.type == "cuda":
        card_guard()
    n_linear = cfg.num_layers - len(cfg.full_attn_layers)
    from .sparse_engine import SparseTracker

    sparse_tracker: SparseTracker | None = None
    draft_num_blocks: int | None = None
    if sparse_k:
        if scorer not in ("bounds", "index"):
            raise ValueError(f"sparse engine scorer {scorer!r}: want bounds|index")
        # The sparse decode/verify tick has its own captured graph
        # (_run_sparse_decode_graph); decode_graph must stay auto so it can be
        # built. The fused sparse read is correct post-#567, so no guard here.
        sparse_tracker = SparseTracker(cfg, sparse_k, scorer, device=backend.device)
        # An explicitly-passed NoPrefixStore means "sharing off" (training/old
        # tests); otherwise the sparse prefix index attaches once the cold tier
        # exists.
        sparse_tracker.sharing_enabled = not isinstance(prefix_store, NoPrefixStore)
        # Device selection (the capture-ready tick) defaults ON whenever a decode
        # graph auto-enables on this backend — its only consumer is the captured
        # sparse tick. A backend _graph_on rejects (CPU, sm70) or explicit
        # decode_graph=False keeps eager; an explicit sparse_device_select wins.
        if sparse_device_select is None:
            sparse_device_select = _graph_on(backend, decode_graph)
        if scorer == "index":
            # materialize ran after construction; keep indexer weights on-device.
            sparse_tracker.iq = sparse_tracker.iq.to(backend.device)
            sparse_tracker.ik = sparse_tracker.ik.to(backend.device)
    if draft is not None:
        draft.set_depth(spec_depth)  # the state pool is sized by the width it settles on
    # materialize moves/rebinds params that change device/dtype; adapters must be added
    # AFTER this (add_lora raises on an unmaterialized model) so they bind to the tensors
    # the forward reads.
    model.params = backend.materialize(model.params)
    model.materialized = True
    # Serve the draft's own weights HERE, before anything reads free memory. They used
    # to be quantized inside Engine.__init__, i.e. after the KV fit had already spent
    # 2/3 of what was free and PrefixStore a quarter of the rest -- so the draft's fp4
    # weights were charged to nothing and `serve --blocks 0 --draft` died in
    # `materialize`'s twiddle with 104 MiB free, never reaching the fit's own print
    # (measured at serve's default --slots 16 on a 32 GB V100). Same ordering fix the
    # state pool below already uses: allocate, then MEASURE what is left.
    if draft is not None:
        _serve_draft(draft, backend)
    # Reclaim the allocator's load-time reserve before anything reads free memory.
    # Loading and quantizing 27B leaves 29.02 GiB reserved against 15.96 allocated on
    # a 31.74 GiB card, and `mem_get_info` counts all 13.06 GiB of that as USED -- so
    # the store's budget below, and the KV fit, see 2.36 GiB free instead of 15.17.
    # Measured on sm70: this one call is the difference between a 1024-token and a
    # 62832-token context at B=1.
    if backend.device.type == "cuda":
        torch.cuda.empty_cache()
    pad = _graph_on(backend, decode_graph)  # the replay's padding row owns a slot and a block
    # The GDN pools first, so fitting the KV pool below can MEASURE what is left
    # instead of estimating it: at slots=3 depth=3 they are 2.94 GiB, 79% of it the
    # per-step verify states, which scale with slots*width and not with max_batch.
    state_pool = LinearStatePool(
        num_slots + pad,
        n_linear,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        device=backend.device,
        dtype=precision.dtype("recurrent_state", backend.device),
        conv_window=cfg.linear_conv_kernel_dim - 1,
        conv_dim=cfg.linear_qkv_dim,
        spec_steps=draft.width if draft is not None else 0,
    )
    # The KV pool's dtype IS the attention kernel's ABI, but only a cuda kernel has one
    # here: main passed NO dtype and every target took PagedKvPool's bf16 default, so
    # routing backend.io in unconditionally moved cpu and metal from bf16 to f32 --
    # K and V stopped being rounded on store on the cell that certifies every kernel in
    # this repo, at 2x the bytes, with the whole suite green (a parity check moves the
    # TileLang and torch sides together, so nothing could see it). io itself is right on
    # cpu; PagedKvPool's DEFAULT is what disagrees with it, so narrow the call site.
    # getattr on BOTH: RefBackend and the other test doubles declare neither, and
    # asking for .arch directly raised AttributeError in 7 tests.
    kv_io = (
        getattr(backend, "io", torch.bfloat16)
        if getattr(backend, "arch", "").startswith("sm")
        else torch.bfloat16
    )
    if kv_fp8 is not None:
        # A fused writer scatters into the plane `kv_layer` returns, which under fp8 is a
        # dequantized copy -- the write is dropped with no error. The fp8 twins take the raw
        # plane plus the scale, so refuse only where a fused writer has no fp8 twin.
        has = getattr(backend, "has_kernel", lambda _n: False)
        blind = [n for n in ("write_tokens", "attn_prep") if has(n) and not has(f"{n}_fp8")]
        if blind:
            raise NotImplementedError(
                f"kv_fp8 with the fused KV writers {blind} on arch "
                f"{getattr(backend, 'arch', '?')}: they scatter into the plane `kv_layer` "
                "returns, which is a dequantized copy under fp8, so every K/V write would be "
                "silently lost, and this cell registers no fp8 twin (docs/design-fp8-kv.md)."
            )
    if sparse_k:
        if not kv_cold_bytes:
            # Cap the host tier at every batch token's page being cold at once.
            # Lazy allocation: this is the LRU ceiling, not bytes reserved.
            from .memory import per_kv_block_bytes

            kv_cold_bytes = (max_total_tokens // BLOCK_TOKENS + 1) * per_kv_block_bytes(
                cfg, kv_io, kv_fp8
            )
        # Per-tick resident peak per slot. Quest selects independently per source
        # group (groups of 4 full-attn layers), and each group's chosen pages
        # co-reside in one shared live map through the forward before finalize
        # demotes them — so selections are a UNION of up to n_groups*k pages, not
        # one k. resolve is idempotent by logical page, so the forced 8-page window
        # and one prefill chunk's own pages add once each even though every group
        # names them. The pool size is the one expression the ledger also prices
        # (memory.sparse_pool_num_blocks), so price and allocate cannot drift.
        from .memory import sparse_pool_num_blocks

        num_blocks = sparse_pool_num_blocks(cfg, num_slots, sparse_k, max_num_batched_tokens)
    elif not num_blocks:
        num_blocks = fit_blocks(
            cfg,
            backend,
            kv_io,
            max_blocks,
            draft_layers=0 if draft is None else draft.cfg.num_layers,
            kv_fp8=kv_fp8,
        )
    # The narrow host-copy dtype. Auto ("") narrows an f32 pool (sm70) to f16 and
    # leaves every other pool native; --cold-format forces it. "native" explicitly
    # keeps the pool dtype even on sm70.
    if cold_format == "native":
        cold_dtype = None
    elif cold_format == "f16":
        cold_dtype = torch.float16
    elif cold_format == "":
        cold_dtype = torch.float16 if kv_io == torch.float32 else None
    else:
        raise ValueError(f"unknown --cold-format {cold_format!r}; expected f16|native")
    kv_pool = PagedKvPool(
        num_blocks + pad,
        cfg.num_kv_heads,
        cfg.head_dim,
        device=backend.device,
        layer_map=cfg.full_attn_layers,
        # Match the attention kernel's IO dtype. sm70's is f32, and a bf16 pool
        # made every attention call cast the WHOLE plane (all num_blocks, not
        # the live ones): 4.71 ms/token, 14% of a 4096-ctx token, independent of
        # context. Same trade the state pool makes below. getattr: test doubles
        # stand in for Backend without declaring an io dtype.
        dtype=kv_io,
        kv_fp8=kv_fp8,
        cold_dtype=cold_dtype,
    )
    if sparse_k and draft is not None:
        # The draft head stays DENSE under sparse, so its pool is independent of the
        # hot set: on a card, fit it to the memory left after weights/state/hot pool
        # (one draft layer is 1/16 of the trunk planes at 27B), capped at a whole
        # context per slot; off cuda the tests are tiny, take the full ceiling.
        from .memory import POOL_FRACTION, draft_per_block_bytes

        cap = num_slots * (max_total_tokens // BLOCK_TOKENS + 1)
        if backend.device.type == "cuda":
            torch.cuda.empty_cache()
            per = draft_per_block_bytes(cfg, kv_io, draft.cfg.num_layers)
            draft_num_blocks = min(
                cap, max(1, int(torch.cuda.mem_get_info()[0] * POOL_FRACTION) // per)
            )
        else:
            draft_num_blocks = cap
    # A resident store entry owns a GDN state snapshot in HBM (144 MiB at 27B f32)
    # and a decode publishes one every BLOCK_TOKENS, so the store's byte budget must
    # fit the card: the 8 GiB default is most of a 32 GB V100's post-weights headroom.
    # Spend a quarter of what is still free after weights and pools.
    kw = {}
    if state_bytes:
        kw["state_bytes"] = state_bytes
    elif backend.device.type == "cuda":
        kw["state_bytes"] = int(torch.cuda.mem_get_info()[0] // 4)
    # Host tier for snapshots the card cannot keep resident. Measured on the live V100: 43
    # of 43 evictions happened with 64% of the block pool free, so every one was state
    # bytes -- a prefix thrown away for a byte the host could hold. 4 GiB rather than the
    # ~25 GiB free: pinned pages cannot be swapped and this pod has 31 GiB of RAM against a
    # 32 GiB card, so pinning most of it destabilises the host, not the process. 4 GiB is 28
    # snapshots against HBM's 9. Not gated on cuda, for the same reason the SSD tier is not:
    # host-to-host is a real demote/promote, and the CPU target is where that is checked.
    if dram_bytes:
        kw["dram"] = DramSnapshots(budget_bytes=dram_bytes)
    if kv_cold_bytes:
        kv_pool.attach_cold(
            HostKvPages(
                budget_bytes=kv_cold_bytes,
                ssd_path=cold_ssd_path,
                ssd_capacity_bytes=cold_ssd_bytes,
            )
        )
    if sparse_k and kv_store:
        raise NotImplementedError(
            "dense bulk boot (--kv-store) with sparse_k>0 is not supported: a boot "
            "load allocates every context block against the device hot pool, which "
            "under sparsity holds only k+window+chunk per slot. Save/boot a dense "
            "build; a sparse-aware boot is a later PR."
        )
    if prefix_store is not None:
        store = prefix_store
    elif sparse_k and not sparse_min_tokens:
        # Pure-sparse cannot use the block-retaining PrefixStore (finalize frees
        # device pages sparse rows do not own); sharing is the tracker's cache.
        store = NoPrefixStore()
    elif sparse_k:
        # Hybrid: dense rows share through the block-retaining store, sparse rows
        # through the tracker's SparsePrefixCache; the per-req sparse_on guard keeps
        # sparse pages out of the dense store, so finalize freeing them is harmless.
        store = PrefixStore(kv_pool, **kw)
    else:
        store = PrefixStore(kv_pool, **kw)
    boot_store = None
    if kv_store:
        # Same fingerprint as the SSD prefix tier: a context computed under other
        # weights or a different kv_fp8 flag must not be bootable.
        boot_store = KvBootStore(kv_store, ssd_fingerprint or _weight_fingerprint(cfg, kv_fp8))
    if sparse_tracker is not None and sparse_tracker.sharing_enabled:
        from .sparse_engine import SparsePrefixCache

        sparse_tracker.prefix = SparsePrefixCache(kv_pool.cold, state_pool)
    return Engine(
        model,
        backend,
        kv_pool,
        state_pool,
        store,
        StepLimits(
            max_batch=max_batch,
            max_total_tokens=max_total_tokens,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        decode_graph=decode_graph,
        draft=draft,
        spec_depth=spec_depth,
        decode=decode,
        sparse_tracker=sparse_tracker,
        sparse_k=sparse_k,
        boot_store=boot_store,
        sparse_device_select=sparse_device_select,
        draft_num_blocks=draft_num_blocks,
        sparse_min_tokens=sparse_min_tokens,
        sparse_prefill_tokens=sparse_prefill_tokens,
    )


def build_serving_engine(
    cfg,
    model,
    backend,
    draft=None,
    depth=2,
    slots=16,
    blocks=0,
    max_ctx=0,
    max_batch=8,
    dram_bytes=0,
    state_bytes=0,
    kv_fp8="",
    decode=None,
    max_batched_tokens=0,
    kv_cold_bytes=0,
    cold_format="",
    cold_ssd_path="",
    cold_ssd_bytes=0,
    sparse_k=DEFAULT_SPARSE_K,
    scorer="bounds",
    kv_store="",
    decode_graph=None,
    sparse_min_tokens=0,
    sparse_prefill_tokens=0,
):
    """Serving-size engine on one card. Multi-card serving is one process per card
    under CUDA_VISIBLE_DEVICES (see generate.py for the process-per-device pattern);
    the in-process DataParallelEngine wrapper was deleted 2026-09-09.

    ``max_ctx`` caps the served context; it still defaults to the model's own limit,
    which for the 27B is 262144 tokens = 275 GB of f32 KV, so it is now a CAP on the
    fit rather than the pool size. ``blocks`` 0 hands the pool to build_engine, which
    fits it after materialize and the allocator reclaim — the only point where free
    memory means anything.

    ``slots`` sizes the GDN state pool; with a draft each slot also owns spec_steps
    of step-state, so a 32 GB card needs 4, not 16.
    """
    ctx = int(max_ctx or cfg.max_position_embeddings)

    kw = dict(
        num_blocks=blocks,
        num_slots=slots,
        max_batch=max_batch,
        max_total_tokens=ctx,
        max_blocks=(ctx * max_batch) // BLOCK_TOKENS,
    )
    if draft is not None:
        kw["draft"], kw["spec_depth"] = draft, depth
    if dram_bytes:
        kw["dram_bytes"] = dram_bytes
    if state_bytes:
        kw["state_bytes"] = state_bytes
    if kv_fp8:
        kw["kv_fp8"] = kv_fp8_dtype(kv_fp8)
    if kv_cold_bytes:
        kw["kv_cold_bytes"] = kv_cold_bytes
    if cold_format:
        kw["cold_format"] = cold_format
    if kv_store:
        kw["kv_store"] = kv_store
    if cold_ssd_path:
        kw["cold_ssd_path"] = cold_ssd_path
        if cold_ssd_bytes:
            kw["cold_ssd_bytes"] = cold_ssd_bytes
    if decode is not None:
        kw["decode"] = decode
    if max_batched_tokens:
        kw["max_num_batched_tokens"] = max_batched_tokens
    # Always forward sparse_k: --sparse-k 0 is the dense opt-out and must reach
    # build_engine to override its sparse default.
    kw["sparse_k"] = sparse_k
    kw["scorer"] = scorer
    kw["decode_graph"] = decode_graph
    kw["sparse_min_tokens"] = sparse_min_tokens
    kw["sparse_prefill_tokens"] = sparse_prefill_tokens
    if sparse_k:
        # Cold pages need somewhere to demote: default the pinned host tier to the whole
        # written context at its per-block bytes if the caller gave no budget.
        kw["kv_cold_bytes"] = kv_cold_bytes or (
            (ctx * max_batch)
            // BLOCK_TOKENS
            * memory.per_kv_block_bytes(
                cfg, torch.bfloat16, kv_fp8_dtype(kv_fp8) if kv_fp8 else None
            )
        )
    return build_engine(cfg, model, backend, **kw)
