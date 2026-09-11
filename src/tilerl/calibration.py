"""Device calibration for the kernel roofline: measured HBM bandwidth and bf16 peak,
appended to the same store the bench views read (``$TILERL_BENCH_STORE`` overrides it).

The roofline divides declared bytes and flops by a MEASURED floor, never a datasheet:
the floor is what THIS card actually does for a large device-to-device copy (HBM
bandwidth) and one big bf16 GEMM (tensor peak), timed with CUDA events. The CPU side of
this module is the ledger read and the bound arithmetic, which is exactly what the gate
asserts; the measurement itself is cuda-only and renders pending-remote off a card."""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

#: one calibration row per (metric, device name) pair
BW_METRIC = "hbm_bw_gbs"
PEAK_METRIC = "bf16_peak_tflops"
FP8_PEAK_METRIC = "fp8_peak_tflops"
PCIE_METRIC = "pcie_h2d_gbs"
#: Pre-Ampere tensor cores (sm70 V100) have no bf16 MMA path; their peak is fp16.
F16_PEAK_METRIC = "f16_peak_tflops"


def store_path() -> Path:
    return Path(
        os.environ.get(
            "TILERL_BENCH_STORE", str(_ROOT / "docs/experience/bench/measurements.jsonl")
        )
    )


def load_rows(path: str | os.PathLike | None = None) -> list[dict]:
    """The ledger's rows via scripts/benchrec (the one reader/writer); a torn tail is
    skipped there. ``path`` swaps benchrec.STORE for the read, so a test store works."""
    from .cli import _benchrec

    br = _benchrec()
    old = br.STORE
    if path is not None:
        br.STORE = Path(path)
    try:
        return br.load_all()
    finally:
        br.STORE = old


def latest_floor(rows: list[dict], metric: str, device_name: str,
                 uuid: str | None = None) -> dict | None:
    """The newest non-superseded calibration ``metric`` row for exactly this device
    name. A row for a different card name is refused by the caller (a V100's bandwidth
    is not an H20's floor) — that is why the match is on the full name, not a prefix.

    Same-name cards are not equally fast (one H20 die measured 8% low at a 1830 vs
    1980 MHz cap), so ``uuid`` (the physical GPU UUID, visible-index independent)
    narrows to that card's own newest row when one exists. When no uuid row exists
    the lookup falls back to the name pool, so pre-uuid rows stay valid floors."""
    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    matches = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("device", {}).get("name") == device_name
        and r.get("id") not in superseded
    ]
    if uuid is not None:
        own = [r for r in matches if r.get("device", {}).get("uuid") == uuid]
        if own:
            matches = own
    return matches[-1] if matches else None


def calibration(rows: list[dict], device_name: str,
                uuid: str | None = None) -> dict | None:
    """The floors for a device, or None when the required bf16 pair is missing (then
    the renderer prints pending-remote rather than dividing by a half-calibration).
    The fp8 peak and the sparse ``pcie_gbs`` floor are both optional: absent → None,
    so fp8/pcie columns render pending rather than borrow the bf16 ceiling. A
    physical ``uuid`` picks that card's own floors when a uuid row exists. The
    tensor peak is bf16 where the arch has it and f16 on pre-Ampere cards (sm70
    V100, no bf16 tensor path); ``peak_metric`` says which one the roofline used."""
    bw = latest_floor(rows, BW_METRIC, device_name, uuid)
    peak = latest_floor(rows, PEAK_METRIC, device_name, uuid)
    peak_metric = PEAK_METRIC
    if peak is None:
        peak = latest_floor(rows, F16_PEAK_METRIC, device_name, uuid)
        peak_metric = F16_PEAK_METRIC
    if bw is None or peak is None:
        return None
    fp8 = latest_floor(rows, FP8_PEAK_METRIC, device_name, uuid)
    pcie = latest_floor(rows, PCIE_METRIC, device_name)
    return {
        "bw_gbs": float(bw["value"]),
        "peak_tflops": float(peak["value"]),
        "peak_metric": peak_metric,
        "fp8_peak_tflops": None if fp8 is None else float(fp8["value"]),
        "pcie_gbs": float(pcie["value"]) if pcie else None,
    }


RESIDENT_METRIC = "device_resident_bytes"
#: the metrics this section renders — a device appears only if it has at least one of
#: these, so an unrelated bench row (e.g. a cpu decode_tok_s) never makes an all-pending
#: section that implies a calibrated card.
_SECTION_METRICS = (BW_METRIC, PEAK_METRIC, FP8_PEAK_METRIC, F16_PEAK_METRIC,
                    PCIE_METRIC, RESIDENT_METRIC)


def device_sections(rows: list[dict]) -> list[dict]:
    """One ledger section per device that has a rendered metric, first-seen order: its
    newest calibration pair and newest residency row, each via :func:`latest_floor`
    (exact device name, superseded skipped). A never-recorded sub-field is None so a
    half-populated card renders pending-remote. Render-only; it never divides by the
    floor."""
    names, seen = [], set()
    for r in rows:
        if r.get("metric") not in _SECTION_METRICS:
            continue
        name = r.get("device", {}).get("name")
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    def pair(r) -> dict | None:
        return None if r is None else {
            "value": r["value"], "commit": r.get("commit"), "date": r.get("date")}

    out = []
    for name in names:
        res = latest_floor(rows, RESIDENT_METRIC, name)
        out.append({
            "device": name,
            "hbm_bw_gbs": pair(latest_floor(rows, BW_METRIC, name)),
            "bf16_peak_tflops": pair(latest_floor(rows, PEAK_METRIC, name)),
            "f16_peak_tflops": pair(latest_floor(rows, F16_PEAK_METRIC, name)),
            "fp8_peak_tflops": pair(latest_floor(rows, FP8_PEAK_METRIC, name)),
            "pcie_h2d_gbs": pair(latest_floor(rows, PCIE_METRIC, name)),
            "residency": None if res is None else {
                "peak": res["value"], "static": res["shape"]["static"],
                "transient": res["shape"]["transient"],
                "commit": res.get("commit"), "date": res.get("date")},
        })
    return out


def bound_seconds(bytes_: int, flops: int, bw_gbs: float, peak_tflops: float) -> float:
    """The roofline lower bound in seconds: max of the byte time and the flop time.
    Pure arithmetic — the one column the CPU gate pins to an exact number."""
    byte_s = bytes_ / (bw_gbs * 1e9)
    flop_s = flops / (peak_tflops * 1e12)
    return max(byte_s, flop_s)


#: faces whose GEMM accumulates on a quant tensor path. The actual MMA dtype is
#: resolved from the kernel registry by launch M (w4a8 fp8 prefill vs bf16 decode);
#: a face here names only that such a resolution is needed.
_QUANT_FACE_NAMES = ("fp8_block_dev", "fp8_dev", "nvfp4", "nvfp4_dev", "nvfp4_dev_b32")


def row_peak_tflops(floors: dict, row: dict, b: int, s: int) -> float | None:
    """The compute ceiling this row's kernel runs AGAINST: the measured peak of the
    MMA dtype the registry kernel issues at this launch's M, not the weight face's.
    nvfp4/fp8 rows with M >= 9 run e4m3 WGMMA (~2x bf16) and use fp8_peak_tflops;
    decode (M<=8) dequants to bf16 and stays on the bf16 peak; nvfp4 uses bf16
    peak. None when the needed floor is absent — the row renders pending rather
    than dividing by the wrong ceiling (a missing fp8 floor on a prefill nvfp4 row
    once read 133.8% against bf16)."""
    from tilerl_kernels.registry import linear_mma_dtype

    from . import precision as P

    face = row.get("face")
    if face is not None and any(getattr(P, n) == face for n in _QUANT_FACE_NAMES):
        op = "linear_fp4" if any(getattr(P, n) == face
                                 for n in ("nvfp4", "nvfp4_dev", "nvfp4_dev_b32")) else "linear_fp8"
        if linear_mma_dtype(op, row_launch_m(row, b, s)) == "fp8":
            return floors.get("fp8_peak_tflops")
    return floors["peak_tflops"]


def _event_seconds(fn, iters: int) -> float:
    """Median wall of ``fn`` over ``iters`` timed with CUDA events (synchronised)."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    times.sort()
    return times[len(times) // 2]


def measure_hbm_bw_gbs(card: int, *, bytes_n: int = 1 << 30, iters: int = 20) -> float:
    """Sustained HBM write+read bandwidth from a >=1 GiB device-to-device copy. A copy
    moves ``bytes_n`` one way; the copy kernel reads the source and writes the dest, so
    divide 2*bytes by the event time — the same HBM-direction rule the ledger uses."""
    import torch

    with torch.cuda.device(card):
        a = torch.empty(bytes_n // 4, dtype=torch.float32, device=f"cuda:{card}")
        b = torch.empty_like(a)
        secs = _event_seconds(lambda: b.copy_(a, non_blocking=True), iters)
    return (2 * bytes_n) / secs / 1e9


def measure_pcie_h2d_gbs(card: int, *, bytes_n: int = 1 << 30, iters: int = 20) -> float:
    """Sustained ONE-WAY host→device bandwidth over PCIe, from a >=1 GiB PINNED host
    tensor copied to the device (the sparse cold-page fetch's path: pageable staging is
    slower, and the fetch uses pinned RAM per #500). Bytes move one way, so divide
    bytes (not 2*bytes) by the event time."""
    import torch

    with torch.cuda.device(card):
        host = torch.empty(bytes_n, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(bytes_n, dtype=torch.uint8, device=f"cuda:{card}")
        secs = _event_seconds(lambda: dev.copy_(host, non_blocking=True), iters)
    return bytes_n / secs / 1e9


def measure_bf16_peak_tflops(card: int, *, n: int = 8192, iters: int = 20) -> float:
    """Sustained bf16 tensor peak from one large square GEMM: 2*n^3 flops / event sec.
    sm70 (V100) has no bf16 tensor path — use measure_f16_peak_tflops there."""
    import torch

    with torch.cuda.device(card):
        a = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}")
        b = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}")
        secs = _event_seconds(lambda: torch.matmul(a, b), iters)
    return (2 * n**3) / secs / 1e12


def measure_f16_peak_tflops(card: int, *, n: int = 8192, iters: int = 20) -> float:
    """Sustained fp16 tensor peak from one large square GEMM. The floor for pre-Ampere
    arches whose only tensor-core path is fp16 (sm70 Volta)."""
    import torch

    with torch.cuda.device(card):
        a = torch.randn(n, n, dtype=torch.float16, device=f"cuda:{card}")
        b = torch.randn(n, n, dtype=torch.float16, device=f"cuda:{card}")
        secs = _event_seconds(lambda: torch.matmul(a, b), iters)
    return (2 * n**3) / secs / 1e12


def measure_fp8_peak_tflops(card: int, *, n: int = 8192, iters: int = 20) -> float:
    """Sustained fp8 tensor peak from one large square GEMM through torch's own fp8
    matmul (scaled e4m3 x e4m3 -> bf16): 2*n^3 flops / event sec. fp8 GEMM rows must
    divide by THIS ceiling, not bf16 peak — fp8 sustains ~2x the bf16 rate, so keying
    an fp8 row to the bf16 floor makes %bound read ~200% on a healthy kernel."""
    import torch

    with torch.cuda.device(card):
        dt = torch.float8_e4m3fn
        a = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}").to(dt)
        b = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}").to(dt)
        # TensorWise scaling: two singleton f32 scales (this torch build rejects 1-D
        # rowwise vectors — it wants [M,1]/[1,N]; the peak GEMM needs only unit scale).
        sa = torch.tensor(1.0, dtype=torch.float32, device=f"cuda:{card}")
        sb = torch.tensor(1.0, dtype=torch.float32, device=f"cuda:{card}")

        def gemm():
            return torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                    out_dtype=torch.bfloat16)

        secs = _event_seconds(gemm, iters)
    return (2 * n**3) / secs / 1e12


def _row(metric: str, value: float, unit: str, device_name: str, card: int,
         derivation: str, uuid: str | None = None, target: str = "sm90"):
    from .cli import _benchrec

    br = _benchrec()
    device = {"name": device_name, "card": card}
    if uuid is not None:
        device["uuid"] = uuid
    return {
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "target": "sm90",
        "build": "eager",
        "model": "device",
        "shape": {"card": card},
        "warm": {"state": "warm", "compiles": None},
        "n": 1,
        "spread": 0,
        "device": device,
        "commit": br.git_commit(),
        "dirty": br.git_dirty(),
        "cmd": f"tilerl bench --calibrate --card {card}",
        "floor": {
            "value": float(value),
            "unit": unit,
            "kind": "measured-best",
            "derivation": derivation,
        },
    }


def device_uuid(card: int) -> str:
    """The physical GPU UUID in nvidia-smi's dashed form, independent of the visible
    ordinal (CUDA_VISIBLE_DEVICES renumbers cards to 0); ties measured rows to the
    die, not to whichever card happened to be masked in."""
    import torch

    return str(torch.cuda.get_device_properties(card).uuid)


def _arch(card: int) -> str:
    """sm-tag of the card (sm70 V100, sm90 H20, ...)."""
    import torch

    major, minor = torch.cuda.get_device_capability(card)
    return f"sm{major}{minor}"


def calibrate_rows(card: int) -> list[dict]:
    """Measure the floors on one card and return the (un-appended) ledger rows.
    sm90+: HBM, bf16 peak, fp8 peak, pinned H2D PCIe. sm70 (V100) has no bf16
    tensor path and no fp8 _scaled_mm path, so it records HBM + the fp16 peak
    only; its fp8/pcie columns render pending rather than a borrowed ceiling.
    Cuda-only; the CLI refuses before calling this off a card."""
    import torch

    device_name = torch.cuda.get_device_name(card)
    uuid = device_uuid(card)
    arch = _arch(card)
    bw = measure_hbm_bw_gbs(card)
    rows = [
        _row(
            BW_METRIC,
            bw,
            "GB/s",
            device_name,
            card,
            "sustained D2D copy >=1 GiB, read+write, CUDA-event median; floor = this measurement",
            uuid,
            target=arch,
        )]
    if arch == "sm70":
        rows.append(_row(
            F16_PEAK_METRIC,
            measure_f16_peak_tflops(card),
            "TFLOP/s",
            device_name,
            card,
            "one large fp16 square GEMM (2n^3 flops), CUDA-event median; sm70 has no bf16 tensor path",
            uuid,
            target=arch))
        return rows
    rows += [
        _row(
            PEAK_METRIC,
            measure_bf16_peak_tflops(card),
            "TFLOP/s",
            device_name,
            card,
            "one large bf16 square GEMM (2n^3 flops), CUDA-event median; floor = this measurement",
            uuid,
            target=arch),
        _row(
            FP8_PEAK_METRIC,
            measure_fp8_peak_tflops(card),
            "TFLOP/s",
            device_name,
            card,
            "one large fp8 (e4m3) scaled square GEMM (2n^3 flops) through torch._scaled_mm, "
            "CUDA-event median; the ceiling for fp8 GEMM rows",
            uuid,
            target=arch),
        _row(
            PCIE_METRIC,
            measure_pcie_h2d_gbs(card),
            "GB/s",
            device_name,
            card,
            "sustained pinned host->device copy >=1 GiB (one-way), CUDA-event median; "
            "the sparse cold-page PCIe fetch floor",
            uuid,
            target=arch),
    ]
    return rows


def append_rows(rows: list[dict], path: str | os.PathLike | None = None) -> list[str]:
    """Validate and append through scripts/benchrec via cli._benchrec, the tree's one
    schema-writer loader. A row failing REQUIRED/unit/target/device/floor checks is
    rejected before the file is opened — calibration is not a second writer. Returns ids."""
    from .cli import _benchrec

    br = _benchrec()
    old = br.STORE
    if path is not None:
        br.STORE = Path(path)
    try:
        return [br.append(r) for r in rows]
    finally:
        br.STORE = old


#: row weight face -> the backend kernel that serves it. The 27B mixes nvfp4 and
#: fp8 linears, so the timed kernel is resolved from the row's OWN face — timing an
#: fp8 row with linear_fp4 would divide fp8 bytes by a different kernel. A face
#: absent here (bf16, fused ops) stays pending rather than running a surrogate.
_FACE_KERNEL = {
    "linear_fp4": ("nvfp4", "nvfp4_dev", "nvfp4_dev_b32"),
    "linear_fp8": ("fp8_block_dev", "fp8_dev"),
}


def resolve_row_kernel(backend, row: dict):
    """The callable serving this roofline row's weight face, or None when the row is
    not a directly-timed quantized GEMM (bf16 row, fused attention/GDN/norms). The
    kernel is chosen by the row's face Format so an fp8 row resolves linear_fp8 and
    an nvfp4 row linear_fp4; never backend.linear (the bf16 surrogate)."""
    from . import precision as P

    face = row.get("face")
    if face is None:
        return None
    kname = next(
        (k for k, names in _FACE_KERNEL.items()
         if any(getattr(P, n) == face for n in names)),
        None,
    )
    if kname is None:
        return None
    fn = getattr(backend, kname, None)
    return fn if callable(fn) else None


def _pack_for(face, w_bf16):
    """(args, kwargs) weight tensors for the kernel the face resolves to. fp4 gets the
    block-32 pack + renorm split (scale + per-row oscale); fp8 gets the [128,128]
    block grid and no oscale (the backend synthesizes its ones grid/row)."""
    from tilerl_kernels import reference

    from . import precision as P

    if face in (P.nvfp4, P.nvfp4_dev, P.nvfp4_dev_b32):
        wq, scale = reference.pack_fp4(w_bf16)
        scale, oscale = reference.renorm_fp4_scale(scale)
        return (wq, scale), {"oscale": oscale}
    w8, wscale = reference.quant_fp8(w_bf16)
    return (w8, wscale), {}


def row_launch_m(row: dict, b: int, s: int) -> int:
    """The M (token rows) of ONE timed launch: decode rows price b*s tokens, lm_head
    prices one vector per sequence (its s=1 weight math repeats per token but it runs
    once after the gather). A decode tick (b rows, s=B per row) and a prefill row
    (b=1, s=S) differ ONLY here — pinning it is what keeps a 1-token GEMM from being
    timed against a full-prefill row."""
    return b if row["name"] == "lm_head" else b * s


def time_row_ms(row: dict, backend, b: int, s: int) -> float | None:
    """ms of the registry kernel the row's face DECLARES, or None to render
    pending-remote. Inputs are packed to that kernel's weight face, so the measured ms
    divides by the same packed bytes the roofline row declares — never a bf16
    surrogate for an nvfp4/fp8 row, and never linear_fp4 for an fp8 row. Fused
    kernels needing engine-shaped inputs return None (card-only probes)."""
    import torch

    if not torch.cuda.is_available():
        return None
    fn = resolve_row_kernel(backend, row)
    if fn is None or row.get("_spec") is None:
        return None
    out_n, inn = tuple(row["_spec"])
    m = row_launch_m(row, b, s)
    dev = backend.device
    x = torch.randn(m, inn, dtype=torch.bfloat16, device=dev)
    w_bf16 = torch.randn(out_n, inn, dtype=torch.bfloat16, device=dev)
    # Shape gate: the built GEMM is exactly the priced launch. flops on a linear row
    # are 2*M*N*K per launch, so the row's declared flops pin M. An m=1 mutant then
    # fails here on a prefill/decode row instead of timing a 1-token GEMM against a
    # full-shape row and printing %bound in the thousands (all timing tests green).
    if row.get("flops") is not None:
        assert row["flops"] == 2 * m * out_n * inn, (
            f"timed M={m} but row prices {row['flops'] // (2 * out_n * inn)} token rows")
    # Pack on CPU: pack_fp4 materializes a nearest-grid LUT of ~8x the weight (a
    # 17408x5120 layer wants 37.9 GiB) which fits a 95 GB H20 but OOMs a 32 GB
    # V100. Packing is untimed fixture prep, so its device/place never enters ms.
    wargs, wkw = _pack_for(row["face"], w_bf16.cpu())
    wargs = tuple(a.to(dev) for a in wargs)
    wkw = {k: v.to(dev) for k, v in wkw.items()}
    # identity assertion: the thing we time is the kernel the row's face declared,
    # not a substitute. resolve_row_kernel is the single resolution point. A bound
    # method (CUDABackend.linear_fp4) forms a NEW wrapper on every getattr, so `is`
    # always fails; compare the underlying function for both bound and static.
    again = resolve_row_kernel(backend, row)
    assert getattr(fn, "__func__", fn) is getattr(again, "__func__", again)
    return _event_seconds(lambda: fn(x, *wargs, **wkw), 1) * 1000.0
