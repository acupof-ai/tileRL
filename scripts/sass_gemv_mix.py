"""Count the sm70 M-path GEMV's SASS instruction mix and REGISTER BUDGET.

**ncu is denied on this pod (ERR_NVGPUCTRPERM) but `nvcc -Xptxas=-v` is not** -- it
reports registers, spills and the occupancy limit with no performance counters at
all. Four indirect A/Bs on task #26 went by before anything in the loop reported a
register count; this is the cheap instrument that should run first.

What it found on the shipped `_xh` M=32 kernel:

    Used 255 registers  ->  255 x 128 threads = ONE block per SM
                        ->  4 of Volta's 64 warps, 6.25% occupancy
    2936 instructions / 1280 HFMA2 = 2.29 per FMA -> issue ceiling 13.6 TFLOPS

Those numbers are facts. The occupancy DIAGNOSIS they suggested is refuted:
`min_blocks=4` buys 4x the warps and M=32 measures 1.00x
(errors/2026-09-03-occupancy-is-not-the-gemv-cap.md). Five mechanisms are now
excluded by measurement -- FMA chain, n_partition, L1 capacity, L1 bandwidth,
occupancy -- so read this script's output as evidence, not as a cause.

tilelang caches every kernel's `device_kernel.cu`, so this recompiles one to read
both numbers statically. The kernel object exposes no cubin path, so the cache
entry is found by grepping for the extern template the variant instantiates --
`_xh` vs not matters, since the f32 path's 1337 F2F casts are a different kernel.

  /usr/bin/python3 -u scripts/sass_gemv_mix.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("TILERL_TARGET", "cuda")
from tilerl_kernels import kernels_linear  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402

MS = (1, 8, 32)
FMA_PEAK = 31.3  # TFLOPS: 80 SM * 64 cores * 2 flop * 1.53 GHz * 2 (half2)
V100_REGS_PER_SM = 65536
V100_WARPS_PER_SM = 64
CACHE = Path.home() / ".tilelang_cache"
#: nvcc needs tilelang's own templates and cutlass's cute headers to rebuild a
#: cached kernel. Both live in the venv/site-packages trees on this pod.
INCLUDES = [
    "**/site-packages/tilelang/src",
    "**/site-packages/flashinfer/data/cutlass/include",
]


def find_includes() -> list[str]:
    out = []
    for pat in INCLUDES:
        hits = sorted(Path("/").glob(pat.lstrip("/")))
        hits += sorted(Path.home().glob(pat))
        if hits:
            out.append(f"-I{hits[0]}")
    return out


def cache_entry(template: str) -> Path | None:
    """The cached device_kernel.cu instantiating `template`.

    The kernel object exposes no cubin path, so the entry is found by the extern
    it instantiates. This has to match the VARIANT exactly: `_xh` and the f32 twin
    are different kernels, and reading the f32 one's 1337 F2F casts as the shipped
    kernel's mix is how the first run of this script went wrong.
    """
    for cu in CACHE.glob("*/*/kernels/*/device_kernel.cu"):
        if template in cu.read_text():
            return cu
    return None


def build(cu: Path) -> tuple[str, int, int]:
    """(sass, registers, spill_store_bytes) for one cached kernel."""
    cubin = Path("/tmp") / (cu.parent.name[:12] + ".cubin")
    p = subprocess.run(["nvcc", "-arch=sm_70", "-cubin", "-Xptxas=-v",
                        *find_includes(), "-o", str(cubin), str(cu)],
                       capture_output=True, text=True, check=False)
    regs = int(m.group(1)) if (m := re.search(r"Used (\d+) registers", p.stderr)) else 0
    spill = int(m.group(1)) if (m := re.search(r"(\d+) bytes spill stores", p.stderr)) else 0
    if not cubin.exists():
        return "", regs, spill
    sass = subprocess.run(["nvdisasm", "-c", str(cubin)], capture_output=True,
                          text=True, check=False).stdout
    return sass, regs, spill


def mix(text: str) -> Counter:
    """Opcode histogram. The address comment and the opcode are separated by a
    long run of whitespace and an optional predicate, so anchoring the opcode to
    the comment (as the first version did) matches almost nothing -- it reported
    0 HFMA2 on a kernel with 1280."""
    out: Counter = Counter()
    for line in text.splitlines():
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if m:
            out[m.group(1).split(".")[0]] += 1
    return out


def main() -> None:
    be = get_backend()
    print(f"# sm70 fp4 GEMV (xh=True, sh=True), FMA peak {FMA_PEAK} TFLOPS")
    for M in MS:
        # Compile so the cache entry exists, then find it by the extern it names.
        kernels_linear.make_linear_fp4_gemv_sm70_m(be.target, M=M, xh=True, sh=True)
        cu = cache_entry(f"tl_fp4_gemv_tiles_f16_m_xh<4,{M}>") or \
            cache_entry(f"tl_fp4_gemv_tiles_f16_m_xh<4, {M}>")
        if cu is None:
            print(f"M={M:>2}: no cache entry naming tl_fp4_gemv_tiles_f16_m_xh<4,{M}>")
            continue
        sass, regs, spill = build(cu)
        blocks = V100_REGS_PER_SM // (regs * 128) if regs else 0
        print(f"\nM={M:>2}: {regs} registers, {spill} B spill stores -> "
              f"{blocks} block/SM = {blocks * 4} of {V100_WARPS_PER_SM} warps "
              f"({100 * blocks * 4 / V100_WARPS_PER_SM:.1f}% occupancy)")
        if not sass:
            print("      (nvcc produced no cubin; see its stderr)")
            continue
        h = mix(sass)
        total, hf = sum(h.values()), h.get("HFMA2", 0)
        if not hf:
            print(f"      {total} instructions, NO HFMA2 -- wrong kernel? "
                  f"top: {h.most_common(5)}")
            continue
        per = total / hf
        print(f"      {total} instructions, {hf} HFMA2, {per:.2f} per FMA "
              f"-> issue ceiling {FMA_PEAK / per:.1f} TFLOPS")
        for op, n in h.most_common(8):
            print(f"      {op:<10} {n:>5} {100 * n / total:>5.1f}%")

    print("\nMeasured M=32 is 17.6% of FMA peak, 35% of the L1-bandwidth ceiling and")
    print("40% of the issue ceiling. Five mechanisms are excluded by measurement")
    print("(FMA chain, n_partition, L1 capacity, L1 bandwidth, occupancy), so these")
    print("numbers are evidence about the kernel, not a diagnosis of the gap.")


if __name__ == "__main__":
    main()
