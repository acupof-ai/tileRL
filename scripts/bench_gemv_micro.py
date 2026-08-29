"""Time the fp4 decode GEMV's (micro_size_k, GROUP) grid. GPU, no model load.

``micro_size_k`` alone sets the weight-stream load width (8 -> LDG.32, 16 -> .64,
32 -> .128); ``micro_size_k * GROUP`` sets the register footprint.
``scripts/_sweep_gemv_micro.py`` already proves all 9 combinations lower, index
exactly and carry no runtime-indexed local array — this is the half that needs a
GPU.

Six arms, not nine. At equal register footprint the widest micro moves the same
bytes in the same shuffle batch with strictly fewer load instructions, so
{(8,4),(16,2),(32,1)} and {(16,4),(32,2)} are ladders, not independent points.
(16,2) rides along as the control ON that dominance argument: same 52 register
slots and same 16 B/thread as (32,1), differing only in load count. If it wins,
the ladder is falsified and all 9 have to be measured.

Weights are **block 16**, which is what the checkpoint ships. Every older sweep
in ``scripts/`` declared block-32 scales and still scored against 0.75 B/elem —
a 1.2x overstatement that puts the *baseline* arm above the kill bar before
anything is tested. Bytes here come from the tensors, never from a constant.

    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src TILERL_TARGET=cuda \
        python3 -u scripts/bench_gemv_micro.py [--compile-only]

``--compile-only`` runs the JIT and the correctness gate and prints no timings —
safe to overlap a model load on the other GPU. Never overlap the timing pass.
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "scripts")
from benchkit import ab, relerr  # noqa: E402

from tilerl_kernels import kernels_linear, reference  # noqa: E402

ARMS = ((8, 4), (32, 1), (32, 2), (32, 4), (16, 1), (16, 2))
SHAPES = (("down", 5120, 17408), ("gate_up", 34816, 5120))
RT, NP, BLOCK = 32, 4, 16  # reduce_thread / n_partition / scale block
#: Marlin's own wall time on down_proj at M=1, the ambition bar (gate B).
#: Its 1.29 TB/s is 0.5625 B/elem (e4m3 group scales); tileRL stores f32 block
#: scales = 0.75 B/elem, so the two TB/s figures are NOT comparable — the
#: comparable quantity is microseconds.
MARLIN_DOWN_US = 38.9


def inputs(n: int, k: int):
    """Random e2m1 nibbles + block-16 scales on device, plus the f32 reference.

    Nibbles are generated directly rather than via ``pack_fp4``, whose ``dist``
    tensor is 32 bytes per weight element (5.7 GB at gate_up).
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    wq = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev, generator=g)
    scale = torch.rand((n, k // BLOCK), device=dev, generator=g) * 0.1 + 0.01
    x = torch.randn((1, k), device=dev, generator=g).bfloat16()
    ref = torch.cat(
        [reference.linear_fp4(x, wq[i : i + 512], scale[i : i + 512]) for i in range(0, n, 512)],
        dim=-1,
    )
    return x, wq, scale, ref


def arm(micro: int, group: int, x, wq, scale):
    k = x.shape[1]
    # _CUDA_PLAN pads this kernel's K to 256; a block covers RT*micro. Direct
    # calls assert instead of padding — see POD-VERIFY, step 8.
    assert k % (RT * micro) == 0, f"K={k} not a multiple of {RT * micro} (micro={micro})"
    fn = kernels_linear.make_linear_fp4_gemv("cuda", micro, group)
    return lambda: (fn(x, wq, scale, RT, NP, BLOCK),)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    ratios: dict[str, list[float]] = {}
    down_us: dict[str, float] = {}
    for label, n, k in SHAPES:
        x, wq, scale, ref = inputs(n, k)
        nbytes = wq.numel() + scale.numel() * 4 + 2 * k
        arms = [(f"micro={m} GROUP={g}", arm(m, g, x, wq, scale)) for m, g in ARMS]
        if args.compile_only:
            for name, fn in arms:
                print(f"{label} {name}: rel-err {relerr(fn()[0], ref):.2e}", flush=True)
            continue
        rows = ab(f"{label} N={n} K={k} — {nbytes / 2**20:.0f} MiB, block {BLOCK}",
                  arms, (ref,), args.iters)
        base = rows[0][1]
        print("\n| arm | ms | TB/s (0.75 B/elem) | vs (8,4) |")
        print("|---|---:|---:|---:|")
        for name, ms, _, ok in rows:
            ratios.setdefault(name, []).append(base / ms if ok else 0.0)
            if label == "down":
                down_us[name] = ms * 1e3
            print(f"| {name} | {ms:.4f} | {nbytes / ms * 1e-9:.3f} | {base / ms:.3f}x |")
    if args.compile_only:
        return

    base_name = f"micro={ARMS[0][0]} GROUP={ARMS[0][1]}"
    won = [n for n, r in ratios.items() if n != base_name and min(r) >= 1.05]
    print("\n== gate A (thesis): >=1.05x vs the shipped arm on BOTH shapes ==")
    print("  PASS: " + ", ".join(won) if won else "  FAIL — memory-level parallelism is not the lever")
    best = min(down_us, key=down_us.get)
    print(f"\n== gate B (ambition): down_proj under {MARLIN_DOWN_US} us, Marlin's wall time ==")
    print(f"  best {best} at {down_us[best]:.1f} us -> {'PASS' if down_us[best] < MARLIN_DOWN_US else 'FAIL'}")
    print("  A pass with a B fail is the expected outcome: widen the load, then")
    print("  move the scale dtype (f32 block scales are 0.25 of tileRL's 0.75 B/elem).")


if __name__ == "__main__":
    main()
