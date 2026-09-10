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

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .kv_cache import BLOCK_TOKENS
from .precision import Format, bf16, f32, kv_format, nbytes

_ROOT = Path(__file__).resolve().parent.parent.parent

#: budget rules build_engine uses (docs/design-cost-model.md); each is the row it makes.
POOL_FRACTION = 2 / 3  # the KV pool takes 2/3 of post-weights free
STATE_FRACTION = 1 / 4  # the resident-snapshot store takes 1/4 of what remains

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


def draft_per_block_bytes(cfg, kv_io, draft_layers: int) -> int:
    """The draft's per-block K+V pool: plain IO dtype, no fp8 scale plane (DraftHead.attach
    builds PagedKvPool without kv_fp8), one pair per draft layer."""
    return draft_layers * nbytes(
        _dtype_fmt(kv_io), (2, cfg.num_kv_heads, BLOCK_TOKENS, cfg.head_dim)
    )


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


def plan(cfg, params: dict | None, device_free: int, *, num_slots: int, num_blocks: int,
         spec_steps: int = 0, state_dtype=f32, kv_io=bf16, kv_fp8=None,
         explicit_state_budget: int = 0, dram_budget: int = 0,
         draft_layers: int = 0, ckpt_faces=None) -> list[Row]:
    """The device rows the engine holds plus the budget rules. ``num_blocks`` is what
    build_engine built (fitted via fit_num_blocks or explicit). Weights come from the
    materialized ``params`` or, for a header-only ``--dry-run --checkpoint`` query, from
    ``ckpt_faces`` (model.checkpoint_weight_faces); both None leaves them pending.
    ``draft_layers>0`` adds the draft pool row (its share of every KV block)."""
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
#: are not allocations and never enter the invariant.
STATIC_OWNERS = ("weights", "state_slots", "kv_pool", "draft_pool")


def static_rows(rows: list[Row]) -> list[Row]:
    """The held allocations in ``rows`` (budget rows excluded). Summed with transient
    they equal the measured peak."""
    return [r for r in rows if r.owner in STATIC_OWNERS]


def transient_bytes(rows: list[Row], peak_bytes: int) -> int:
    """The resident bytes that are NOT one of the named static allocations — activation
    scratch, fragmented allocator blocks, captured-graph side buffers. The invariant the
    ledger prints is ``peak = sum(static) + transient``; transient is derived as the
    residual, never allocated here."""
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

    Returns (rows, totals): one dict per static allocation and budget rule, in plan order,
    with tier / owner / derived / measured / delta, followed by a final ``transient`` row
    so ``sum(static) + transient == measured peak``, and per-tier totals over the HELD
    allocations (static + transient; budget rows are excluded from the total).

    ``peak_bytes`` None (never measured — no forward has run, or off cuda with no peak)
    suppresses the transient row rather than printing a guess.
    """
    out: list[dict] = []
    held_total = 0
    for r in plan_rows:
        held = r.owner in STATIC_OWNERS
        m = measured.get(r.owner)
        if held:
            held_total += r.n
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
    totals: dict[str, int] = {}
    if peak_bytes is not None:
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
        held_total += transient
    # per-tier total over held allocations only
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
    return out, totals


def residency_row(
    device_name: str,
    card: int | None,
    peak_bytes: int,
    static_bytes: int,
    transient_bytes: int,
    target: str,
    model: str = "27B-nvfp4",
    build: str = "eager",
):
    """One ledger row recording steady-state device residency and its static/transient
    split, so occupancy lives in the same measurements.jsonl as the kernel roofline.
    The shape carries both halves of ``peak = static + transient``. ``target`` is the
    benchrec target (sm90/sm70/cpu/metal = backend.arch) and ``card`` the physical GPU;
    a card-less sm* row is benchrec-rejected, so the CLI refuses --record-residency
    off cuda rather than fabricating one. Appended through scripts/benchrec (the one
    schema writer), never written directly here."""

    def _git(args):
        try:
            return subprocess.run(
                ["git", *args], cwd=str(_ROOT), capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _git(["rev-parse", "HEAD"]) or "unknown"
    dirty = bool(_git(["status", "--porcelain"]))
    return {
        "metric": "device_resident_bytes",
        "value": int(peak_bytes),
        "unit": "bytes",
        "target": target,
        "build": build,
        "model": model,
        "shape": {
            "card": card if card is not None else 0,
            "static": int(static_bytes),
            "transient": int(transient_bytes),
        },
        "warm": {"state": "warm", "compiles": None},
        "n": 1,
        "spread": 0,
        "device": {"name": device_name, "card": card},
        "commit": commit,
        "dirty": dirty,
        "cmd": "tilerl serve --dry-run --record-residency",
        "floor": {
            "value": int(peak_bytes),
            "unit": "bytes",
            "kind": "reference",
            "derivation": f"measured resident peak = static {int(static_bytes)} + "
            f"transient {int(transient_bytes)}",
        },
    }


def append_residency(row: dict, path: str | None = None) -> str:
    """Validate + append through scripts/benchrec (the tree's one schema writer)."""
    import importlib.util

    br_path = _ROOT / "scripts" / "benchrec.py"
    spec = importlib.util.spec_from_file_location("tilerl_benchrec", br_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if path is not None:
        mod.STORE = Path(path)
    return mod.append(row)


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
