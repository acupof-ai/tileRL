"""Pre-pod sweep of the fp4 decode GEMV's (micro_size_k, GROUP) grid, without a GPU or nvcc:
does each combination lower, what WQ load width it gets, register array sizes, and exact index
arithmetic (full coverage, one scale per scale block). CUDA C via scripts/cuda_codegen.enable().

    uv run python scripts/_sweep_gemv_micro.py
"""

from __future__ import annotations

import re
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cuda_codegen import TARGET, enable  # noqa: E402
from tilerl_kernels import reference  # noqa: E402

MICRO = (8, 16, 32)
GROUPS = (1, 2, 4)
RT = 32  # reduce_thread; the backend hardcodes 32
NP = 4  # n_partition
#: f32-equivalent register slots per element, by emitted C type.
_SLOT = {"float": 1.0, "nv_bfloat16": 0.5, "uchar": 0.25}
_VEC = {"uint": 4, "uint2": 8, "uint4": 16}


# ---------------------------------------------------------------- codegen probe


def _arrays(src: str) -> dict[str, tuple[str, int]]:
    """Per-thread local arrays: name -> (C type, element count). Scalars dropped."""
    found = re.findall(r"\n  (float|uchar|nv_bfloat16) (\w+)\[(\d+)\];", src)
    return {n: (t, int(s)) for t, n, s in found if int(s) > 1}


def _runtime_indexed(src: str, names) -> list[str]:
    # a runtime-indexed register array falls to local memory (cost 22% of roof once)
    unrolled = set(re.findall(r"#pragma unroll\n\s*for \(int (\w+)", src))
    runtime = (set(re.findall(r"for \(int (\w+)", src)) - unrolled) | {"threadIdx"}
    bad = []
    for name in names:
        for idx in re.findall(rf"\b{name}\[([^\]]*)\]", src):
            if runtime & set(re.findall(r"[A-Za-z_]\w*", idx)):
                bad.append(f"{name}[{idx}]")
    return bad


def probe(micro: int, group: int, n: int, k: int, block: int = 16) -> dict:
    from tilerl_kernels import kernels_linear as kl

    row: dict = {"micro": micro, "GROUP": group}
    try:
        src = kl.make_linear_fp4_gemv(TARGET, micro, group).get_kernel_source(
            torch.zeros(1, k, dtype=torch.bfloat16),
            torch.zeros(n, k // 2, dtype=torch.uint8),
            torch.zeros(n, k // block),
            RT,
            NP,
            block,
        )
    except Exception as e:  # noqa: BLE001 — a combination that dies here is dead for the pod
        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:110]
        return row
    arrays = _arrays(src)
    vecs = set(re.findall(r"= \*\((\w+)\*\)\(WQ ", src))
    assert len(vecs) == 1, f"mixed WQ load widths {vecs}"
    vec = vecs.pop()
    row |= {
        "src": src,
        "wq_vec": vec,
        # one micro-tile is one vector load (micro/2 <= 16 B); GROUP per iter
        "wq_loads": group * (micro // 2) // _VEC[vec],
        "wq_bytes": micro // 2 * group,
        "arrays": arrays,
        "slots": sum(_SLOT[t] * s for t, s in arrays.values()),
        "spill": _runtime_indexed(src, arrays),
    }
    return row


# ---------------------------------------------------------------- index / numeric


def segments(micro: int, group: int, k: int, block: int):
    """Replay make_linear_fp4_gemv's schedule in issue order: (k0, length, scale index) per segment."""
    block_k = RT * micro
    num_ko = -(-k // block_k)
    num_g = num_ko // group
    nseg = max(1, micro // block)
    seg = micro // nseg
    tiles = [kg * group + g for kg in range(num_g) for g in range(group)]
    tiles += [num_g * group + kt for kt in range(num_ko - num_g * group)]
    for kr in range(RT):
        for ko in tiles:
            base = ko * block_k + kr * micro
            for s in range(nseg):
                yield base + s * seg, seg, (base + s * seg) // block


def check_index(micro: int, group: int, k: int, block: int) -> str:
    """Every k touched exactly once, in range, under the scale of its own block."""
    seen = [0] * k
    for k0, ln, sidx in segments(micro, group, k, block):
        if k0 + ln > k:
            return f"OVER-READ k={k0 + ln - 1} >= K={k}"
        for kk in range(k0, k0 + ln):
            seen[kk] += 1
            if sidx != kk // block:
                return f"WRONG SCALE at k={kk}: kernel {sidx} != {kk // block}"
    if min(seen) != 1 or max(seen) != 1:
        return f"COVERAGE {min(seen)}..{max(seen)} per element (want 1)"
    return "ok"


def check_numeric(micro: int, group: int, block: int, unsegmented: bool = False) -> float:
    """Kernel's f32 accumulation order vs reference.linear_fp4; unsegmented=True must FAIL for micro > block."""
    torch.manual_seed(7)
    n, k = 4, RT * micro * (2 * group + 1)  # the odd tile forces the K-tail loop
    w = torch.randn(n, k)
    x = torch.randn(1, k)
    wq, scale = reference.pack_fp4(w, block)
    grid = reference.dequant_fp4(wq, torch.ones_like(scale))  # unscaled e2m1 values
    xf, gf, sf = x[0].float(), grid.float(), scale.float()
    y = torch.zeros(n)
    for k0, ln, sidx in segments(micro, group, k, block):
        if unsegmented:
            if k0 % micro:
                continue  # only the first segment's scale survives, over the whole tile
            ln = micro
        y += sf[:, sidx] * (xf[k0 : k0 + ln] * gf[:, k0 : k0 + ln]).sum(-1)
    ref = reference.linear_fp4(x, wq, scale)[0]
    return ((y - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()


def check_dequant_math() -> str:
    """All 16 nibbles of the kernel's lut (_e2m1_fp32) vs reference.dequant_fp4."""
    got = []
    for nib in range(16):
        e, m = (nib >> 1) & 3, nib & 1
        bits = (((nib & 8) << 28) | ((126 + e) << 23) | ((m << 22) & -min(e, 1))) & -min(e | m, 1)
        bits = (bits & 0xFFFFFFFF) - (1 << 32) * (bits >> 31 & 1)  # to signed int32
        got.append(torch.tensor([bits], dtype=torch.int32).view(torch.float32).item())
    wq = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
    want = reference.dequant_fp4(wq, torch.ones(1, 1))[0].tolist()  # nibbles 0..15 in order
    bad = [(i, got[i], want[i]) for i in range(16) if got[i] != want[i]]
    return "ok" if not bad else f"MISMATCH {bad}"


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    enable()
    fail = 0
    print(f"dequant math (_e2m1_fp32 vs reference.dequant_fp4): {check_dequant_math()}")

    shapes = {"down (N=5120 K=17408)": (5120, 17408), "gate_up (N=34816 K=5120)": (34816, 5120)}
    for name, (n, k) in shapes.items():
        print(f"\n== {name}, block=16, reduce_thread=32, n_partition=4 ==")
        print(
            f"{'micro':>5} {'GRP':>4} {'lower':>6} {'WQ load':>8} {'B/thr/it':>9} {'loads/it':>9}"
            f" {'regs':>6}  {'local arrays':<26} {'spill':>6}  index"
        )
        for micro in MICRO:
            for group in GROUPS:
                r = probe(micro, group, n, k)
                if "error" in r:
                    print(f"{micro:>5} {group:>4} {'NO':>6}  {r['error']}")
                    fail += 1
                    continue
                idx = check_index(micro, group, k, 16)
                arrays = " ".join(f"{a}[{s}]" for a, (_, s) in r["arrays"].items())
                print(
                    f"{micro:>5} {group:>4} {'yes':>6} {r['wq_vec']:>8} {r['wq_bytes']:>9}"
                    f" {r['wq_loads']:>9} {r['slots']:>6.0f}  {arrays:<26}"
                    f" {('YES' if r['spill'] else 'no'):>6}  {idx}"
                )
                fail += bool(r["spill"]) + (idx != "ok")

    print("\n== K padding: _CUDA_PLAN pads this kernel's K to 256 ==")
    for micro in MICRO:
        need = RT * micro
        idx = check_index(micro, 1, 17664, 16)  # down's 17408, rounded to a 256 multiple
        print(f"  micro={micro:>2}: K must be a multiple of {need:>4}; K=17664 -> {idx}")
        fail += (idx != "ok") != (need > 256)

    print("\n== numeric: kernel accumulation order vs reference.linear_fp4 ==")
    for block in (16, 32):
        for micro in MICRO:
            for group in GROUPS:
                err = check_numeric(micro, group, block)
                fail += err >= 1e-2
                print(
                    f"  block={block} micro={micro:>2} GROUP={group}:"
                    f" max rel err {err:.2e} {'ok' if err < 1e-2 else 'FAIL'}"
                )

    print("\n== negative control: one scale per micro-tile (the un-segmented schedule) ==")
    for micro in MICRO:
        err = check_numeric(micro, 1, 16, unsegmented=True)
        wrong = err > 1e-2
        print(f"  block=16 micro={micro:>2}: max rel err {err:.2e} -> {'WRONG' if wrong else 'exact'}")
        fail += wrong != (micro > 16)  # must be wrong exactly when the tile spans blocks

    print(f"\n{'FAILURES: ' + str(fail) if fail else 'all checks pass'}")
    sys.exit(1 if fail else 0)
