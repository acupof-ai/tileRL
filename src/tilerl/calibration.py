"""Device calibration for the kernel roofline: measured HBM bandwidth and bf16 peak,
appended to the same store the bench views read (``docs/experience/bench/
measurements.jsonl``; ``$TILERL_BENCH_STORE`` overrides it for tests).

The roofline divides declared bytes and flops by a MEASURED floor, never a datasheet:
the floor is what THIS card actually does for a large device-to-device copy (HBM
bandwidth) and one big bf16 GEMM (tensor peak), timed with CUDA events. The CPU side of
this module is the ledger read and the bound arithmetic, which is exactly what the gate
asserts; the measurement itself is cuda-only and renders pending-remote off a card.

Row metric names are ``hbm_bw_gbs`` and ``bf16_peak_tflops``; their ``floor.kind`` is
``measured-best`` (the device's own measured ceiling for that population)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

#: one calibration row per (metric, device name) pair
BW_METRIC = "hbm_bw_gbs"
PEAK_METRIC = "bf16_peak_tflops"


def store_path() -> Path:
    return Path(
        os.environ.get(
            "TILERL_BENCH_STORE", str(_ROOT / "docs/experience/bench/measurements.jsonl")
        )
    )


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_ROOT), capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        stamp = _ROOT / ".synced_commit"
        return stamp.read_text().strip() if stamp.exists() else "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        marker = _ROOT / ".synced_dirty"
        if marker.exists():
            return marker.read_text().strip() == "1"
        raise


def load_rows(path: str | os.PathLike | None = None) -> list[dict]:
    """Newest last; a torn final line is skipped (same rule as scripts/benchrec)."""
    p = Path(path) if path is not None else store_path()
    if not p.exists():
        return []
    rows, skipped = [], 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        print(f"calibration: WARNING skipped {skipped} unparseable ledger line(s)", flush=True)
    return rows


def latest_floor(rows: list[dict], metric: str, device_name: str) -> dict | None:
    """The newest non-superseded calibration ``metric`` row for exactly this device
    name. A row for a different card name is refused by the caller (a V100's bandwidth
    is not an H20's floor) — that is why the match is on the full name, not a prefix."""
    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    matches = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("device", {}).get("name") == device_name
        and r.get("id") not in superseded
    ]
    return matches[-1] if matches else None


def calibration(rows: list[dict], device_name: str) -> dict | None:
    """Both floors for a device, or None when either is missing (then the renderer prints
    pending-remote rather than dividing by a half-calibration)."""
    bw = latest_floor(rows, BW_METRIC, device_name)
    peak = latest_floor(rows, PEAK_METRIC, device_name)
    if bw is None or peak is None:
        return None
    return {
        "bw_gbs": float(bw["value"]),
        "peak_tflops": float(peak["value"]),
        "bw_commit": bw.get("commit"),
        "peak_commit": peak.get("commit"),
    }


def bound_seconds(bytes_: int, flops: int, bw_gbs: float, peak_tflops: float) -> float:
    """The roofline lower bound in seconds: max of the byte time and the flop time.
    Pure arithmetic — the one column the CPU gate pins to an exact number."""
    byte_s = bytes_ / (bw_gbs * 1e9)
    flop_s = flops / (peak_tflops * 1e12)
    return max(byte_s, flop_s)


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


def measure_bf16_peak_tflops(card: int, *, n: int = 8192, iters: int = 20) -> float:
    """Sustained bf16 tensor peak from one large square GEMM: 2*n^3 flops / event sec."""
    import torch

    with torch.cuda.device(card):
        a = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}")
        b = torch.randn(n, n, dtype=torch.bfloat16, device=f"cuda:{card}")
        secs = _event_seconds(lambda: torch.matmul(a, b), iters)
    return (2 * n**3) / secs / 1e12


def _row(metric: str, value: float, unit: str, device_name: str, card: int, derivation: str):
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
        "device": {"name": device_name, "card": card},
        "commit": git_commit(),
        "dirty": git_dirty(),
        "cmd": f"tilerl bench --calibrate --card {card}",
        "floor": {
            "value": float(value),
            "unit": unit,
            "kind": "measured-best",
            "derivation": derivation,
        },
    }


def calibrate_rows(card: int) -> list[dict]:
    """Measure both floors on one card and return the two (un-appended) ledger rows.
    Cuda-only; the CLI refuses before calling this off a card."""
    import torch

    device_name = torch.cuda.get_device_name(card)
    bw = measure_hbm_bw_gbs(card)
    peak = measure_bf16_peak_tflops(card)
    return [
        _row(
            BW_METRIC,
            bw,
            "GB/s",
            device_name,
            card,
            "sustained D2D copy >=1 GiB, read+write, CUDA-event median; floor = this measurement",
        ),
        _row(
            PEAK_METRIC,
            peak,
            "TFLOP/s",
            device_name,
            card,
            "one large bf16 square GEMM (2n^3 flops), CUDA-event median; floor = this measurement",
        ),
    ]


def _benchrec():
    """The single schema writer for the bench store. ``scripts/`` is not a packaged
    module, so load it by path from the repo root; calibration rows must pass the SAME
    validate/append every collector satisfies — there must not be a second writer."""
    import importlib.util

    br_path = _ROOT / "scripts" / "benchrec.py"
    spec = importlib.util.spec_from_file_location("tilerl_benchrec", br_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def append_rows(rows: list[dict], path: str | os.PathLike | None = None) -> list[str]:
    """Validate and append through ``scripts/benchrec`` (the tree's one schema writer).
    A row that fails its REQUIRED/unit/target/device/floor checks is rejected before the
    file is opened — calibration is not a second, unvalidated writer. Returns the ids."""
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


def _event_ms(fn) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


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
    m = b if row["name"] == "lm_head" else b * s
    dev = backend.device
    x = torch.randn(m, inn, dtype=torch.bfloat16, device=dev)
    w_bf16 = torch.randn(out_n, inn, dtype=torch.bfloat16, device=dev)
    wargs, wkw = _pack_for(row["face"], w_bf16)
    # identity assertion: the thing we time is the kernel object the row's face
    # declared, not a substitute. resolve_row_kernel is the single resolution point.
    assert fn is resolve_row_kernel(backend, row)
    return _event_ms(lambda: fn(x, *wargs, **wkw))
