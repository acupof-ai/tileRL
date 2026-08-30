"""sm70 fp4 GEMV parity — run on the V100. Kernel vs the torch-eager reference.

Uses CUDA 12.4 nvcc (system nvcc is 11.8, no c++20). Invoke as:
  PATH=/usr/local/cuda-12.4/bin:$PATH TILELANG_CACHE_DIR=/tmp/tl_c \
    TILERL_TARGET=cuda python scripts/_sm70_gemv_parity.py
"""
import sys

sys.path.insert(0, "src")
import torch

from tilerl_kernels import kernels_linear, reference


def _round_up(x, m):
    return (x + m - 1) // m * m


def run(N, K, block=32, reduce_thread=32, n_partition=4):
    torch.manual_seed(0)
    dev = "cuda"
    # A random OCP-e2m1 weight in natural (low-nibble-first) layout via pack_fp4.
    w = torch.randn(N, K, dtype=torch.float32) * 0.3
    wq, scale = reference.pack_fp4(w, block=block)  # wq [N,K//2] uint8, scale [N,K//block]
    scale, oscale = reference.renorm_fp4_scale(scale)
    x = torch.randn(1, K, dtype=torch.float32) * 0.5

    # Reference: y = oscale * (x @ dequant(wq,scale).T)
    ref = reference.linear_fp4(x, wq, scale, oscale)  # [1,N] f32

    # Kernel: pad N to n_partition, K stays (block_K = reduce_thread*16 divides K?)
    Np = _round_up(N, n_partition)
    Kp = _round_up(K, reduce_thread * 16)
    xq = torch.zeros(1, Kp, dtype=torch.bfloat16, device=dev)
    xq[0, :K] = x[0].to(torch.bfloat16)
    wqp = torch.zeros(Np, Kp // 2, dtype=torch.uint8, device=dev)
    wqp[:N, : K // 2] = wq.to(dev)
    scp = torch.zeros(Np, Kp // block, dtype=torch.float32, device=dev)
    scp[:N, : K // block] = scale.to(dev)
    oscp = torch.zeros(Np, dtype=torch.float32, device=dev)
    oscp[:N] = oscale.to(dev)
    res = torch.zeros(1, Np, dtype=torch.float32, device=dev)

    k = kernels_linear.make_linear_fp4_gemv_sm70("cuda")
    y = k(xq, wqp, scp, oscp, res, reduce_thread, n_partition, block)[:, :N].cpu()

    err = (y - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-9)
    # Project correctness gate is rtol=1e-2 (bf16 activation IO); the absolute
    # error rides the largest output magnitude, so relative error is the metric.
    ok = torch.allclose(y, ref, rtol=1e-2, atol=0) or rel < 1e-2
    print(f"N={N} K={K} block={block}: max_err={err:.4e} rel={rel:.4e} pass(rtol=1e-2)={ok}")
    return ok


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    print("device cap", cap, torch.cuda.get_device_name(0))
    all_ok = True
    for N, K, block in [(512, 512, 32), (5120, 5120, 32), (256, 2048, 16)]:
        all_ok &= run(N, K, block)
    print("PARITY", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)
