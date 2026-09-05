"""Where does an 11K prefill token go on sm70: GDN forward, attention, or linears?

sm70 has no `gdn_state_scan` (verified via `_resolve`), so `_gdn_chunk_wy` falls into
its else branch and the GDN core is `reference.gdn_chunk_core` -- a torch bmm loop over
64-token chunks. 48 of 64 layers are GDN. If that loop dominates, it is a cost every
request pays, unlike a prefix cache that only pays off on a hit.

Uses torch.profiler over ONE prefill forward of the real model at the real shape, and
attributes by kernel/op name rather than by wrapping layers -- wrapping changes what is
timed, and an eager microbench of a single layer carries a launch floor that swamps the
answer (docs/experience: the ~60us eager launch floor).

Prints the top ops by self CUDA time plus a GDN / attention / linear rollup. One job at
a time: reads nvidia-smi first and refuses if another process holds the card.
"""

import os
import subprocess
import sys

import torch


def card_busy(mine: int) -> str | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    others = [ln for ln in out.splitlines() if ln.strip()
              and int(ln.split(",")[0]) != mine]
    return "\n".join(others) or None


def main() -> None:
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    busy = card_busy(os.getpid())
    if busy:
        print(f"card busy, refusing:\n{busy}")
        return

    from tilerl_kernels.backend import _resolve, get_backend

    from tilerl.cli import _build_model

    backend = get_backend()
    ks = _resolve(backend.precision, backend.arch)
    print(f"arch {backend.arch} io {backend.io} "
          f"gdn_state_scan={'gdn_state_scan' in ks} "
          f"gdn_chunk_fused={'gdn_chunk_fused' in ks}")

    cfg, model = _build_model("qwen38-27b", 0, fuse_projections=True)
    from tilerl.engine import build_engine

    engine = build_engine(cfg, model, backend, num_blocks=0, num_slots=8,
                          max_batch=1, max_total_tokens=t + 64, prefix_store=None,
                          max_blocks=(t + 64) * 3 // 16, decode_graph=False)
    from tilerl.engine import SamplingParams

    ids = [3 + (i * 7) % 100000 for i in range(t)]
    # One prefill tick, warmed once so JIT is not in the window.
    engine.submit(list(ids), SamplingParams(max_new_tokens=1, seed=0))
    engine.step()
    torch.cuda.synchronize()

    engine.submit(list(ids), SamplingParams(max_new_tokens=1, seed=1))
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        engine.step()
        torch.cuda.synchronize()

    evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in evs)
    print(f"\ntotal self CUDA {total / 1000:.1f} ms over T={t} "
          f"= {total / 1000 / t:.3f} ms/token\n")
    print(f"{'op':<52}{'self ms':>10}{'%':>7}{'calls':>8}")
    for e in sorted(evs, key=lambda e: -e.self_device_time_total)[:18]:
        print(f"{e.key[:52]:<52}{e.self_device_time_total / 1000:>10.2f}"
              f"{e.self_device_time_total / total * 100:>7.1f}{e.count:>8}")

    # Rollup: bmm/baddbmm and triangular solve are the WY loop's own ops.
    groups = {
        "GDN (bmm/solve/cumsum/exp)": ("bmm", "baddbmm", "triangular", "cumsum",
                                       "exp", "sigmoid", "silu"),
        "attention (paged/dense)": ("attention", "paged", "softmax"),
        "linear (fp4/fp8 gemm)": ("linear", "gemm", "gemv", "matmul", "mm"),
    }
    print()
    for name, pats in groups.items():
        ms = sum(e.self_device_time_total for e in evs
                 if any(p in e.key.lower() for p in pats)) / 1000
        print(f"{name:<32}{ms:>9.2f} ms{ms / (total / 1000) * 100:>7.1f}%"
              f"{ms / t:>9.3f} ms/token")
    print("\nnote: groups overlap by substring; the per-op table above is authoritative.")


if __name__ == "__main__":
    main()
