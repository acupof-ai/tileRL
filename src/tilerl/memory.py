"""The memory ledger: every byte the engine HOLDS is derived in ONE place.

``plan`` is the allocator's INPUT, not a second computation of a number the pool also
computes: the fit in build_engine calls :func:`fit_num_blocks`, the KV pool row calls
:func:`per_kv_block_bytes`, and LinearStatePool's tensors are the shapes in
:func:`state_shapes`. A CPU test that merely shows the two agree today cannot stop them
drifting tomorrow, so there is only one formula for each number.

GPU-measured columns are pending-remote; the budget arithmetic is not — ``plan`` takes
``device_free`` as a parameter, so on CPU you pass the number and the free*2/3 pool
fit and free/4 snapshot budget rows are exercised without a card. On the CPU tiny model
the weights, KV pool and state rows are exact (tests/test_memory_ledger.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .kv_cache import BLOCK_TOKENS
from .precision import Format, bf16, f32, kv_format, nbytes

#: budget rules build_engine uses (docs/design-cost-model.md); each is the row it makes.
POOL_FRACTION = 2 / 3  # the KV pool takes 2/3 of post-weights free
STATE_FRACTION = 1 / 4  # the resident-snapshot store takes 1/4 of what remains

#: A training micro-step segments every layer once its rows are longer than this;
#: mirrors train._MLP_SEGMENT_MAX_T so the plan and _step pick the same tape shape.
MLP_SEGMENT_MAX_T = 1280

_DTYPE_FMT = {16: bf16, 32: f32}


def _dtype_fmt(dtype) -> Format:
    return dtype if isinstance(dtype, Format) else _DTYPE_FMT.get(dtype.itemsize * 8, bf16)


@dataclass(frozen=True)
class Row:
    """One allocation or budget."""

    tier: str  # device | host | ssd
    owner: str
    n: int
    note: str = ""


def per_kv_block_bytes(cfg, kv_io, kv_fp8=None) -> int:
    """Bytes of ONE trunk KV block: K+V over every full-attn plane, the fp8 per-head_dim
    scale plane included. MAIN pool only — the draft's pool is a separate pool and a
    separate row (:func:`draft_per_block_bytes`); folding it in here would count it twice
    (once in kv_pool, once in draft_pool)."""
    planes = 2 * len(cfg.full_attn_layers)
    io_fmt = _dtype_fmt(kv_io)
    kv_fmt = kv_format(cfg.head_dim) if kv_fp8 is not None else io_fmt
    return nbytes(kv_fmt, (planes, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim))


# --- sparse KV (docs/design-sparse-kv.md) -----------------------------------
def sparse_source_count(cfg) -> int:
    """Independent source groups — the SAME count group_map uses to size the
    pool. Tiny (<4 full-attn) is one group per plane; otherwise the planes split
    into sparse_index.INDEX_SOURCE_LAYERS groups. Delegates to the one canonical
    sparse_group_count so the ledger never prices n_full//4 (wrong for, e.g.,
    8 full-attn planes: 4 groups, not 2)."""
    from .sparse_index import sparse_group_count

    return sparse_group_count(len(cfg.full_attn_layers))


def sparse_pages(context_tokens: int) -> int:
    """Whole 16-token pages a context occupies (window subtraction assumes full pages)."""
    return -(-int(context_tokens) // BLOCK_TOKENS)


def _index_key_format(di: int = 128) -> Format:
    """One projected page key per index head: ``di`` fp8 elements + one f32
    scale per key (nbytes di+4). The shipped face is di=128 (132 B); the tiny
    cell uses di=16 (20 B)."""
    return Format(bits=8, scales=((di, "f32"),))


def index_keys_bytes(cfg, pages: int, di: int = 128) -> int:
    """Learned-indexer keys resident on device: pages x source layers x index heads,
    one [di] fp8 key (di+4 B) each. Shipped di=128 on the 27B: pages x 4 x 4 x
    132 = 2,112 B/page = 132 B/token, 33.0 MiB at 256k. The live engine passes
    its actual di so the tiny cell (di=16) reconciles derived == measured."""
    from .sparse_index import INDEX_HEADS

    ih = min(INDEX_HEADS, cfg.num_kv_heads)
    per = nbytes(_index_key_format(di), (di,))
    return pages * sparse_source_count(cfg) * ih * per


def page_bounds_bytes(cfg, pages: int) -> int:
    """Training-free Quest bounds resident on device: per page, per full-attn layer,
    per KV head, a (kmin,kmax) pair over head_dim in fp16. 27B: 64 KiB/page =
    4 KiB/token."""
    return pages * 2 * len(cfg.full_attn_layers) * cfg.num_kv_heads * cfg.head_dim * 2


def sparse_hot_pages_per_slot(cfg, k_pages: int, max_num_batched_tokens: int) -> int:
    """Resident hot pages one slot's pool is sized for — the exact expression
    build_engine allocates: ``n_groups*k + WINDOW + chunk_pages``. Groups choose
    independently, so the worst case is their DISJOINT union (n_groups*k); the
    forced own window and one max-size prefill chunk (+1 partial) co-reside.
    Canonical here so the ledger row and the pool cannot drift apart."""
    from .sparse_index import WINDOW_PAGES

    chunk_pages = max_num_batched_tokens // BLOCK_TOKENS + 1
    return sparse_source_count(cfg) * k_pages + WINDOW_PAGES + chunk_pages


def sparse_pool_num_blocks(cfg, num_slots: int, k_pages: int,
                           max_num_batched_tokens: int) -> int:
    """Total device blocks build_engine allocates for a sparse engine:
    one per slot's hot ceiling plus a single shared spare block."""
    return num_slots * sparse_hot_pages_per_slot(cfg, k_pages, max_num_batched_tokens) + 1


def sparse_rows(cfg, *, num_rows: int, num_slots: int, context_tokens: int,
                k_pages: int, max_num_batched_tokens: int,
                scorer: str, kv_io, kv_fp8=None) -> list[Row]:
    """The three sparse-KV owners (design-sparse-kv.md "Cost model rows"):

    - ``index_keys`` (learned scorer) or ``page_bounds`` (Quest scorer) on device;
    - ``kv_hot``: the device pool build_engine ALLOCATES — ``num_slots`` times the
      per-slot hot ceiling ``n_groups*k + 8-window + chunk`` plus one spare block.
      This is a capacity row (bounded by worst-case disjoint groups), not the number
      of a context's pages that happen to be resident this tick;
    - ``kv_cold`` on host: per concurrent row, every written page outside the
      selected + window hot set (the chunk pages are transient write headroom, not
      part of the context's cold set).
    """
    from .sparse_index import WINDOW_PAGES

    if scorer not in ("index", "bounds"):
        raise ValueError(f"unknown sparse scorer {scorer!r}; want index|bounds")
    written = sparse_pages(context_tokens)
    pool_blocks = sparse_pool_num_blocks(cfg, num_slots, k_pages, max_num_batched_tokens)
    selected_hot = min(sparse_source_count(cfg) * k_pages + WINDOW_PAGES, written)
    cold_pages = written - selected_hot
    block = per_kv_block_bytes(cfg, kv_io, kv_fp8)
    scorer_n = index_keys_bytes(cfg, written) if scorer == "index" \
        else page_bounds_bytes(cfg, written)
    return [
        Row("device", "index_keys" if scorer == "index" else "page_bounds", scorer_n,
            f"{written} written pages, {scorer} scorer, {k_pages} hot + {WINDOW_PAGES} window"),
        Row("device", "kv_hot", pool_blocks * block,
            f"{pool_blocks} allocated pool blocks "
            f"({num_slots} slots x {sparse_hot_pages_per_slot(cfg, k_pages, max_num_batched_tokens)} hot ceiling + 1 spare)"),
        Row("host", "kv_cold", num_rows * cold_pages * block,
            f"{num_rows} rows x {cold_pages} cold pages"),
    ]



def draft_per_block_bytes(cfg, kv_io, draft_layers: int) -> int:
    """The draft's per-block K+V pool: plain IO dtype, no fp8 scale plane (DraftHead.attach
    builds PagedKvPool without kv_fp8), one pair per draft layer."""
    return draft_layers * nbytes(
        _dtype_fmt(kv_io), (2, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim)
    )


def per_cold_kv_block_bytes(cfg, kv_io, kv_fp8=None, cold_dtype=None) -> int:
    """Bytes ONE demoted page occupies in the host/SSD tier, mirroring
    ``PagedKvPool._page_blob`` plane for plane. K/V take the narrow cold dtype when
    set (sm70 narrows its f32 pool to f16); otherwise the pool's own storage dtype
    (fp8 payload when kv_fp8 is on, else kv_io). The fp8 per-head_dim scale planes
    always stay f32 and are priced separately — kv_format() already bundles them, so
    the fp8-native branch uses it as-is and must NOT add them again (cc caught a
    1792-vs-1280 double count on tiny fp8)."""
    planes = 2 * len(cfg.full_attn_layers)
    shape = (planes, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim)
    if cold_dtype is None and kv_fp8 is not None:
        return nbytes(kv_format(cfg.head_dim), shape)
    # narrow cold dtype (f16 on an f32 pool; fp8 is refused on sm70 so this is
    # plain K/V), or a plain native pool: payload only ...
    payload_fmt = _dtype_fmt(cold_dtype) if cold_dtype is not None else _dtype_fmt(kv_io)
    total = nbytes(payload_fmt, shape)
    if kv_fp8 is not None:
        # ... plus the two f32 per-token scale planes a narrow/plain payload omits.
        total += 2 * planes * cfg.num_kv_heads * BLOCK_TOKENS * 4
    return total


def fit_num_blocks(
    cfg, device_free: int, kv_io, kv_fp8=None, draft_layers: int = 0, cap: int = 0, floor: int = 64
) -> int:
    """The KV blocks that fit ``device_free`` under the pool rule. The fit's denominator is
    main + draft per block (the draft pool mirrors num_blocks), but the fitted count feeds
    BOTH pools — it is one number. Pure arithmetic: on CPU the caller passes device_free."""
    per_block = per_kv_block_bytes(cfg, kv_io, kv_fp8) + draft_per_block_bytes(
        cfg, kv_io, draft_layers
    )
    n = max(floor, int(device_free * POOL_FRACTION) // per_block)
    return min(n, cap) if cap else n


def reserve_kv_blocks(
    *, base_kv_blocks: int, kv_per_block: int, idle_free_bytes: int, reserve_bytes: int
) -> tuple[int, int]:
    """KV blocks to cut so the build leaves at least ``reserve_bytes`` free.

    Pure arithmetic over ONE pre-pool snapshot: ``idle_free_bytes`` is the free
    the finished layout is PROJECTED to have (measured free after weights/state,
    minus the KV, draft and prefix-store bytes about to be allocated), not a
    post-allocation measurement. The pool is then built once at the returned
    count — never allocated full and resized (that needs new+old to coexist and
    OOMs in the exact low-free condition this guards). The graph pad block is
    already counted by the caller in the projected KV bytes. A floor already met
    cuts 0 blocks. Returns ``(final_kv_blocks, dropped_blocks)``.

    Raises when even an empty KV pool cannot leave the floor — the card is too
    small for the fixed weights/state plus the reserve asked for.

    This is a PEAK-LIVE reduction, not a held reservation: nothing pins the free
    bytes against the caching allocator reclaiming them after a large transient.
    """
    if reserve_bytes < 0:
        raise ValueError(f"device reserve must be >= 0, got {reserve_bytes}")
    shortfall = reserve_bytes - idle_free_bytes
    if shortfall <= 0:
        return base_kv_blocks, 0
    kv_total = base_kv_blocks * kv_per_block
    if reserve_bytes > idle_free_bytes + kv_total:
        raise ValueError(
            f"device reserve {reserve_bytes} bytes cannot be met: projected idle free is "
            f"{idle_free_bytes} and the whole KV pool is only {kv_total} bytes; "
            f"{reserve_bytes - idle_free_bytes - kv_total} bytes short even with 0 KV blocks"
        )
    drop = min(base_kv_blocks, math.ceil(shortfall / kv_per_block))
    return base_kv_blocks - drop, drop


def layout_with_reserve(
    *, free0_bytes: int, reserve_bytes: int, base_kv_blocks: int, kv_per_block: int,
    pad_blocks: int = 0, draft_blocks: int = 0, draft_per_block: int = 0,
    other_static_bytes: int = 0,
) -> tuple[int, int]:
    """Size the KV pool ONCE from a pre-pool free snapshot with a reserve floor.

    ``free0_bytes`` is measured after the fixed weights + GDN state pool exist but
    BEFORE any KV/draft/prefix tensor; the caller has already frozen the draft
    block count and the prefix-store byte budget, passed here as already-known
    bytes. Projected idle free at the BASE sizing is
    ``free0 - (base+pad)*kv_per - draft*per - other_static``. Only the trimmable
    KV blocks shrink to meet the floor (the pad and the frozen draft/store sizing
    do not), so the returned pool is allocated directly — never built full and
    resized. Returns ``(final_kv_blocks, dropped_blocks)``. reserve_bytes 0 is a
    no-op and returns the base unchanged, which is why build keeps its old path
    when the reserve is off.
    """
    if reserve_bytes <= 0:
        return base_kv_blocks, 0
    kv_bytes = (base_kv_blocks + pad_blocks) * kv_per_block
    draft_bytes = draft_blocks * draft_per_block
    idle = free0_bytes - kv_bytes - draft_bytes - other_static_bytes
    return reserve_kv_blocks(
        base_kv_blocks=base_kv_blocks,
        kv_per_block=kv_per_block,
        idle_free_bytes=idle,
        reserve_bytes=reserve_bytes,
    )


def state_shapes(cfg, num_slots: int, spec_steps: int = 0) -> list[tuple[tuple, object]]:
    """The (shape, dtype-kind) of every LinearStatePool tensor — the one list both the pool
    and the state rows price. kind 'state' uses the pool storage dtype, 'parity' is int32."""
    n_lin = cfg.num_layers - len(cfg.full_attn_layers)
    nvh, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    conv_k = cfg.linear_conv_kernel_dim - 1
    out: list[tuple[tuple, object]] = [((num_slots, n_lin, nvh, dk, dv), "state")]
    if n_lin and conv_k:
        out.append(((num_slots, n_lin, 2, conv_k, cfg.linear_qkv_dim), "state"))
    if spec_steps and n_lin:
        out.append(((num_slots, n_lin, spec_steps, nvh, dk, dv), "state"))
        if conv_k:
            out.append(((num_slots, n_lin, spec_steps, conv_k, cfg.linear_qkv_dim), "state"))
    out.append(((num_slots,), "parity"))
    return out


def _state_bytes(cfg, num_slots: int, state_fmt: Format, spec_steps: int = 0) -> int:
    return sum(
        nbytes(f32 if kind == "parity" else state_fmt, shape)
        for shape, kind in state_shapes(cfg, num_slots, spec_steps)
    )


def weight_row(params: dict) -> Row:
    """Served weights summed from the materialized tensors. A quantized model holds
    .wq/.scale/.oscale sidecars (and deletes the bf16 master), so this is the storage
    truth, not param_specs — the gate asserts this equals the live params."""
    return Row("device", "weights", sum(t.numel() * t.element_size() for t in params.values()))


def weight_row_faces(faces: dict) -> Row:
    """The weights row from :func:`model.checkpoint_weight_faces` — every served key's
    (shape, device face), names mapped and bf16 fp4-keys repacked exactly as load_hf.
    This equals load_hf's resident bytes to the integer (27B gate), which the raw
    headers cannot: they include non-served tensors and price the repacked keys on disk."""
    return Row("device", "weights", sum(nbytes(fmt, shape) for shape, fmt in faces.values()))


# --- training rows ----------------------------------------------------------
def lora_shapes(specs: dict, rank: int) -> list[tuple]:
    """The [rank,K] A / [N,rank] B adapter pair for every base linear add_lora
    attaches to. Mirrors model.add_lora's attachment rule: a 2-D served linear,
    excluding conv1d (the canonical param_specs keys carry no quant sidecars)."""
    out = []
    for key, shape in specs.items():
        if len(shape) != 2 or key.endswith("conv1d"):
            continue
        n, k = shape
        out.append((rank, k))
        out.append((n, rank))
    return out


def adapter_row(specs: dict, rank: int, fmt: Format = bf16) -> Row:
    """The LoRA adapter tensors (precision role 'adapter', bf16)."""
    return Row("device", "adapter", sum(nbytes(fmt, shp) for shp in lora_shapes(specs, rank)),
               f"rank {rank}")


def adamw_state_row(specs: dict, rank: int | None = None, *, adapter: bool,
                    fmt: Format = f32) -> Row:
    """AdamW moments: two f32 tensors the shape of every trained param. With
    ``adapter`` the trained set is the LoRA pairs; otherwise it is every param."""
    shapes = lora_shapes(specs, rank) if adapter else list(specs.values())
    return Row("device", "optimizer_state", 2 * sum(nbytes(fmt, shp) for shp in shapes),
               "adamw m+v")


def adafactor_state_row(specs: dict, fmt: Format = f32, *, iso: bool = False) -> Row:
    """Adafactor's factored state per 2-D trained param (one f32 row + one f32
    column) plus one full f32 state per non-2-D param. beta1=0 so there is no
    first-moment tensor.

    Under ISO the 2-D trained tensors are the FRAMES U [N,r] and V [K,r], not
    the weights: Adafactor holds a factored (row+column) pair per frame, so each
    2-D weight contributes N+r + K+r factors."""
    total = 0
    for shp in specs.values():
        if len(shp) != 2:
            total += nbytes(fmt, shp)
            continue
        n, k = shp
        if iso:
            r = min(n, k)
            total += nbytes(fmt, (n,)) + nbytes(fmt, (r,)) + nbytes(fmt, (k,)) + nbytes(fmt, (r,))
        else:
            total += nbytes(fmt, (n,)) + nbytes(fmt, (k,))
    return Row("device", "optimizer_state", total,
               "adafactor factored v over ISO frames" if iso else "adafactor factored v")


def iso_frame_row(specs: dict, fmt: Format = f32) -> Row:
    """The ISO frames of every 2-D trained param: U [N,r], S [r], V [K,r] f32
    (r=min(N,K)). Full-parameter SFT wraps Adafactor, whose own state is a
    separate optimizer_state row; on CUDA these frames are host-resident."""
    total = 0
    for n, k in (s for s in specs.values() if len(s) == 2):
        r = min(n, k)
        total += nbytes(fmt, (n, r)) + nbytes(fmt, (r,)) + nbytes(fmt, (k, r))
    return Row("host", "frame", total, "ISO U,S,V per 2-D weight")


def _tape_segments(cfg, b: int, s: int) -> int:
    """One segment boundary row per layer when the rows are long (segment='layer'
    in train._step); below the threshold MLPs alone are checkpointed and their
    live activations stay on the tape (not priced by this boundary formula)."""
    return cfg.num_layers if s > MLP_SEGMENT_MAX_T else 0


def tape_row(cfg, b: int, s: int, *, lora_rank: int | None) -> Row:
    """What one recorded train step's tape holds after the forward, in the
    segment='layer' mode the 27B uses (B*S > 1280): the bf16 embedding lookup,
    one f32 boundary hidden per layer, the final-norm hidden, and the head output
    [b,s,v] (the logits fed to the loss grad) — always present in training since
    the head is scored even when its weight is tied to the embedding. With an
    adapter the head also keeps two more [b,s,v] tensors (frozen-base output,
    residual add) plus its A delta [b,s,r].

    Derived from counting RecordingBackend's real entries on the tiny model
    (tests pin it); nothing else stays live — every in-layer activation is
    recomputed from its segment input during backward."""
    h, v = cfg.hidden_size, cfg.vocab_size
    total = nbytes(bf16, (b, s, h))  # embedding lookup
    # f32 h-rows that survive the layer segments: one boundary per layer, plus the
    # final norm output (the 2-mlp-ckpt rows at short S do not exist in this mode).
    total += (_tape_segments(cfg, b, s) + 1) * nbytes(f32, (b, s, h))
    total += nbytes(f32, (b, s, v))  # head output / logits (training always scores it)
    if lora_rank is not None:
        # the head adapter keeps its A delta [b,s,r], its B output, the frozen-base
        # output and the residual add — 3 [b,s,v] total with the logits above.
        # True on a tied head too: add_lora attaches at the shared embed/head base.
        total += 2 * nbytes(f32, (b, s, v)) + nbytes(f32, (b, s, lora_rank))
    return Row("device", "tape", total, f"B{b} x S{s} layer-segment activations")


def train_plan(cfg, b: int, s: int, *, lora_rank: int | None = None,
               optim: str = "adamw") -> list[Row]:
    """The rows the TRAINING engine holds for one step, separate from the serving
    :func:`plan` (weights plus the rows here). ``lora_rank`` set = LoRA RL/OPD:
    adapter bf16 + AdamW moments over the adapter pairs. None = full SFT: every
    param is trainable, ``optim`` selects adamw, adafactor, or iso (adafactor
    state plus host-resident f32 frames)."""
    from .model import param_specs

    specs = param_specs(cfg)
    rows: list[Row] = []
    if lora_rank is not None:
        rows.append(adapter_row(specs, lora_rank))
        rows.append(adamw_state_row(specs, lora_rank, adapter=True))
    elif optim == "adamw":
        rows.append(adamw_state_row(specs, adapter=False))
    else:
        rows.append(adafactor_state_row(specs, iso=optim == "iso"))
        if optim == "iso":
            rows.append(iso_frame_row(specs))
    rows.append(tape_row(cfg, b, s, lora_rank=lora_rank))
    return rows


def plan(cfg, params: dict | None, device_free: int, *, num_slots: int, num_blocks: int,
         spec_steps: int = 0, state_dtype=f32, kv_io=bf16, kv_fp8=None,
         explicit_state_budget: int = 0, dram_budget: int = 0,
         draft_layers: int = 0, ckpt_faces=None,
         sparse: dict | None = None) -> list[Row]:
    """The device rows the engine holds plus the budget rules. ``num_blocks`` is what
    build_engine built (fitted via fit_num_blocks or explicit). Weights come from the
    materialized ``params`` or, for a header-only ``--dry-run --checkpoint`` query, from
    ``ckpt_faces`` (model.checkpoint_weight_faces); both None leaves them pending.
    ``draft_layers>0`` adds the draft pool row (its share of every KV block).
    ``sparse`` = {'num_rows','context','k_pages','scorer'} adds the three sparse-KV rows
    and replaces the dense kv_pool as the held pool (kv_hot is the device subset)."""
    rows: list[Row] = []
    if ckpt_faces is not None:
        rows.append(weight_row_faces(ckpt_faces))
    elif params is not None:
        rows.append(weight_row(params))
    rows.append(
        Row(
            "device",
            "state_slots",
            _state_bytes(cfg, num_slots, _dtype_fmt(state_dtype), spec_steps),
        )
    )
    if sparse is not None:
        # Sparse deployment: the device pool IS the pinned hot set (plus scorer keys);
        # cold pages live on host. No dense kv_pool row to avoid counting hot twice.
        # Sparse forces the decode graph off, so there is NO replay pad slot: price
        # the hot pool for the spec's own slot count, not the (pad-inflated) num_slots
        # the state row above correctly uses for the dense/graph path.
        rows.extend(sparse_rows(
            cfg, num_rows=sparse["num_rows"], num_slots=sparse["num_slots"],
            context_tokens=sparse["context"], k_pages=sparse["k_pages"],
            max_num_batched_tokens=sparse["max_num_batched_tokens"],
            scorer=sparse["scorer"], kv_io=kv_io, kv_fp8=kv_fp8))
    else:
        rows.append(
            Row(
                "device",
                "kv_pool",
                per_kv_block_bytes(cfg, kv_io, kv_fp8) * num_blocks,
                f"{num_blocks} blocks",
            )
        )
    if draft_layers:
        rows.append(
            Row(
                "device",
                "draft_pool",
                draft_per_block_bytes(cfg, kv_io, draft_layers) * num_blocks,
                f"{draft_layers} draft layers x {num_blocks} blocks",
            )
        )
    # Budget rules as *_budget rows so a reader summing allocations skips them.
    if device_free:
        rows.append(
            Row("device", "kv_pool_budget", int(device_free * POOL_FRACTION), "rule free*2/3")
        )
        rows.append(
            Row(
                "device",
                "prefix_entries_budget",
                explicit_state_budget or int(device_free * STATE_FRACTION),
                "rule free/4",
            )
        )
    if dram_budget:
        rows.append(Row("host", "prefix_entries_budget", dram_budget, "dram tier budget"))
    return rows


#: owners that are held allocations (in peak = Σ static + transient). Budget-rule rows
#: are not allocations and never enter the invariant. The training rows are held too;
#: they never appear in a serving :func:`plan`, so they cannot enter its peak residual.
STATIC_OWNERS = (
    "weights", "state_slots", "kv_pool", "draft_pool",
    "index_keys", "page_bounds", "kv_hot", "sparse_graph",
    "adapter", "optimizer_state", "frame", "tape",
)

#: HELD allocations on the host/SSD tier. They enter a {tier}_total but never the
#: device ``peak = Σ static + transient`` invariant (that equation is device-only;
#: kv_cold lives in pinned host RAM while its device frame is freed).
HELD_HOST_OWNERS = frozenset({"kv_cold", "kv_cold_ssd", "kv_prefix"})


def static_rows(rows: list[Row]) -> list[Row]:
    """Held allocations; budget rows are excluded from peak = Σ static + transient."""
    return [r for r in rows if r.owner in STATIC_OWNERS]


def transient_bytes(rows: list[Row], peak_bytes: int) -> int:
    """The resident bytes that are NOT one of the named static allocations — activation
    scratch, fragmented allocator blocks, captured-graph side buffers. The invariant the
    ledger prints is ``peak = sum(static) + transient``; transient is derived as the
    residual, never allocated here. STATIC_OWNERS is device-tier only; a host/ssd held
    owner (HELD_HOST_OWNERS: kv_cold) is not on the card and does not subtract."""
    held = sum(r.n for r in static_rows(rows))
    t = int(peak_bytes) - held
    if t < 0:
        raise ValueError(
            f"measured peak {peak_bytes} is below the {held} bytes of static allocations; "
            "the peak was taken before the pools were built or a static row is missing"
        )
    return t


def memory_table(plan_rows: list[Row], measured: dict[str, int], peak_bytes: int | None):
    """ONE presentation of the device ledger, shared by ``serve --dry-run`` and /health.

    One dict per static allocation and budget rule, in plan order, with tier / owner /
    kind / derived / measured / delta, followed by a final ``transient`` row so
    ``sum(static) + transient == measured peak``, and a ``{tier}_total`` row per tier
    over the HELD allocations (budget rows excluded from the total).

    ``peak_bytes`` None (never measured — no forward has run, or off cuda with no peak)
    suppresses the transient row and totals rather than printing a guess.
    """
    out: list[dict] = []
    for r in plan_rows:
        held = r.owner in STATIC_OWNERS or r.owner in HELD_HOST_OWNERS
        m = measured.get(r.owner)
        out.append(
            {
                "tier": r.tier,
                "owner": r.owner,
                "kind": "allocation" if held else "budget",
                "derived": r.n,
                "note": r.note,
                "measured": m,
                "delta": (r.n - m) if m is not None else None,
            }
        )
    if peak_bytes is None:
        return out
    transient = transient_bytes(plan_rows, peak_bytes)
    out.append(
        {
            "tier": "device",
            "owner": "transient",
            "kind": "allocation",
            "derived": transient,
            "note": "peak - Σ static",
            "measured": transient,
            "delta": 0,
        }
    )
    # per-tier total over held allocations (static + transient), budget rows excluded
    totals: dict[str, int] = {}
    for r in out:
        if r["kind"] == "allocation":
            totals[r["tier"]] = totals.get(r["tier"], 0) + r["derived"]
    for tier, total in totals.items():
        out.append(
            {
                "tier": tier,
                "owner": f"{tier}_total",
                "kind": "total",
                "derived": total,
                "note": "held allocations (static + transient)",
                "measured": None,
                "delta": None,
            }
        )
    return out


def format_memory_table(rows: list[dict]) -> str:
    """The single text rendering both surfaces print; :func:`memory_table` is the data."""
    lines = [f"  {'owner':<24} {'derived MiB':>12} {'measured MiB':>13} {'delta':>7}  note"]
    for r in rows:
        if r["kind"] == "total":
            lines.append(f"  {'-' * 24} {'-' * 12}")
            lines.append(
                f"  {r['owner']:<24} {r['derived'] / 2**20:12.2f} {'':>13} {'':>7}  {r['note']}"
            )
            continue
        m = f"{r['measured'] / 2**20:13.2f}" if r["measured"] is not None else f"{'-':>13}"
        d = f"{r['delta']:7d}" if r["delta"] is not None else f"{'-':>7}"
        lines.append(f"  {r['owner']:<24} {r['derived'] / 2**20:12.2f} {m} {d}  {r['note']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Measured ledger builders (moved out of Engine, step 10b). Pure: every value
# they read is an argument - no Engine import, no live-counter aggregation.
# _build_stats stays on Engine and passes explicit values/callables.
# ---------------------------------------------------------------------------


def held_storage(*, cfg, model_params, kv, states, sparse=None, sparse_min_tokens=0,
                 draft_kv=None, running=(),
                 sparse_graph_bytes: Callable[[], int] = lambda: 0) -> dict[str, int]:
    """Measured bytes of each static owner from materialized tensor storage. The
    ledger's measured column and the CPU peak (off cuda there is no torch
    allocator high-water). Pool sums include both fp8 scale grids: dropping them
    makes derived != measured on the fp8 27B path by two planes."""

    def pool(p) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (p.k_pool, p.v_pool, p.k_scale, p.v_scale)
            if t is not None
        )

    measured = {
        "weights": sum(t.numel() * t.element_size() for t in model_params.values()),
        "kv_pool": pool(kv),
        "state_slots": sum(
            t.numel() * t.element_size()
            for t in (
                states.states,
                states.conv_windows,
                states.step_states,
                states.step_windows,
                states.win_parity,
            )
            if t is not None
        ),
    }
    # Hybrid mode reconciles the DENSE ledger (the pool is sized for dense rows
    # pinning their whole context); sparse-only measured owners (kv_hot/bounds/
    # cold) are omitted - the pool is one container for both modes.
    if sparse is not None and sparse_min_tokens == 0:
        block_n = per_kv_block_bytes(cfg, kv.dtype, kv.kv_fp8)
        hot = sum(len(sparse.resident.get(r.req_id, ())) for r in running) * block_n
        measured.pop("kv_pool")
        measured["index_keys" if sparse.scorer == "index" else "page_bounds"] = (
            sparse.bounds_bytes()
        )
        measured["kv_hot"] = hot
        if getattr(kv, "cold", None) is not None:
            shared_n = kv.cold.shared_bytes()
            if kv.cold.bytes_held - shared_n:
                measured["kv_cold"] = kv.cold.bytes_held - shared_n
            if shared_n:
                measured["kv_prefix"] = shared_n
        graph_n = sparse_graph_bytes()
        if graph_n:
            measured["sparse_graph"] = graph_n
    if draft_kv is not None:
        measured["draft_pool"] = pool(draft_kv)
    return measured


def measured_peak_bytes(backend) -> int | None:
    """The resident-device byte peak the table closes against. On cuda this is
    the torch allocator's high-water mark; off cuda None - the Engine sums the
    held-storage dict itself for its CPU peak (it already holds that dict)."""
    import torch

    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(backend.device))
    return None


def memory_rows(*, cfg, model_params, kv, states, held: dict[str, int],
                peak_bytes: int | None, sparse=None, sparse_min_tokens=0,
                running=(), draft_kv=None, draft_layers: int = 0, boot=None,
                sparse_graph_bytes: Callable[[], int] = lambda: 0,
                sparse_graph_count: Callable[[], int] = lambda: 0) -> list[dict]:
    """The unified device ledger rows (memory_table input): every held allocation
    plus the transient residual, so ``measured peak = sum static + transient``.
    Pure: per-call plan/table; the Engine owns any static-ledger memoization."""
    # Spec step planes live in the SAME pool as states (step_states/step_windows,
    # sized by draft.width + 1 at build). Read the count off the real tensor: a
    # hardcoded 0 priced 776 MiB where the pool holds 2273 (d1 hybrid), dropping
    # both spec planes and making every device-byte judgment that reads the ledger
    # undercount GDN state by ~3x.
    spec_steps = 0 if states.step_states is None else states.step_states.shape[2]
    if sparse is None or sparse_min_tokens:
        derived = plan(
            cfg,
            model_params,
            0,
            num_slots=states.num_slots,
            num_blocks=kv.num_blocks,
            spec_steps=spec_steps,
            state_dtype=states.states.dtype,
            kv_io=kv.dtype,
            kv_fp8=kv.kv_fp8,
            draft_layers=draft_layers,
        )
    else:
        # Live sparse ledger from ACTUAL per-row state (post-tick every private
        # page is demoted, so kv_hot counts the blocks currently in r.blocks).
        derived = plan(
            cfg,
            model_params,
            0,
            num_slots=states.num_slots,
            num_blocks=kv.num_blocks,
            spec_steps=spec_steps,
            state_dtype=states.states.dtype,
            kv_io=kv.dtype,
            kv_fp8=kv.kv_fp8,
            draft_layers=0,
            sparse=None,
        )
        derived = [r for r in derived if r.owner != "kv_pool"]
        block_n = per_kv_block_bytes(cfg, kv.dtype, kv.kv_fp8)
        # Sparse draft pool is dense-sized (a whole context per slot), unlike the
        # hot trunk pool, so its row is computed from the draft pool's blocks.
        if draft_kv is not None:
            derived.append(
                Row(
                    "device",
                    "draft_pool",
                    draft_per_block_bytes(cfg, kv.dtype, draft_layers) * draft_kv.num_blocks,
                    f"{draft_layers} draft layers x {draft_kv.num_blocks} dense blocks",
                )
            )
        pages_total = 0
        hot_n = 0
        for r in running:
            complete = r.seq_len // BLOCK_TOKENS
            pages_total += complete
            hot_n += len(r.blocks) * block_n
        if sparse.scorer == "index":
            scorer_n = index_keys_bytes(cfg, pages_total, sparse.di)
            owner = "index_keys"
            note = f"{pages_total} complete pages, learned index scorer"
        else:
            # Actual stored bounds, not complete-page count: the newest complete
            # page's bound is written in finalize, so a snapshot between forwards
            # sees one fewer bound than seq_len//16.
            scorer_n = sparse.bounds_bytes()
            owner = "page_bounds"
            note = f"{pages_total} complete pages, bounds scorer"
        cold_n = ssd_n = ssd_cap = prefix_n = 0
        if getattr(kv, "cold", None) is not None:
            prefix_n = kv.cold.shared_bytes()
            cold_n = kv.cold.bytes_held - prefix_n
            ssd_n = kv.cold.ssd_bytes
            ssd_cap = kv.cold.ssd_capacity_bytes
        derived.append(Row("device", owner, scorer_n, note))
        derived.append(Row("device", "kv_hot", hot_n, "resident private blocks this tick"))
        if cold_n:
            derived.append(Row("host", "kv_cold", cold_n, "demoted pages in pinned host RAM"))
        if prefix_n:
            derived.append(Row("host", "kv_prefix", prefix_n,
                               "shared published-prefix pages in pinned host RAM"))
        if ssd_n:
            derived.append(Row("ssd", "kv_cold_ssd", ssd_n,
                               "demoted pages spilled past the host budget"))
        if ssd_cap:
            derived.append(Row("ssd", "kv_cold_ssd_capacity", ssd_cap,
                               "countable spill budget used by admission"))
        graph_n = sparse_graph_bytes()
        if graph_n:
            derived.append(
                Row(
                    "device",
                    "sparse_graph",
                    graph_n,
                    f"captured sparse decode tick staging, {sparse_graph_count()} bucket(s)",
                )
            )
    rows = memory_table(derived, held, peak_bytes)
    # Dense engine with a cold tier driven by the manual sparse_retier seam (#500):
    # its host pages are not in plan(), so append the held allocation explicitly.
    cold = getattr(kv, "cold", None)
    if sparse is None and cold is not None and (cold.bytes_held or cold.ssd_bytes):
        per = per_cold_kv_block_bytes(cfg, kv.dtype, kv.kv_fp8, kv.cold_dtype)
        st = cold.stats()
        if cold.bytes_held:
            pages = st["kv_cold_pages"]
            rows.append(
                {
                    "tier": "host",
                    "owner": "kv_cold",
                    "kind": "allocation",
                    "derived": pages * per,
                    "note": f"{pages} demoted pages in host RAM",
                    "measured": cold.bytes_held,
                    "delta": pages * per - cold.bytes_held,
                }
            )
        if cold.ssd_bytes:
            pages = st["kv_cold_ssd_pages"]
            rows.append(
                {
                    "tier": "ssd",
                    "owner": "kv_cold_ssd",
                    "kind": "allocation",
                    "derived": pages * per,
                    "note": f"{pages} demoted pages spilled to SSD",
                    "measured": cold.ssd_bytes,
                    "delta": pages * per - cold.ssd_bytes,
                }
            )
    # The cold-start boot store on disk: derived == measured bytes (one
    # kv_cold(ssd) allocation per saved entry), so plan and filesystem agree.
    if boot is not None:
        ssd_bytes = boot.bytes_total()
        if ssd_bytes:
            rows.append(
                {
                    "tier": "ssd",
                    "owner": "kv_cold",
                    "kind": "allocation",
                    "derived": ssd_bytes,
                    "note": f"{boot.entries()} saved boot entries",
                    "measured": ssd_bytes,
                    "delta": 0,
                }
            )
    return rows
