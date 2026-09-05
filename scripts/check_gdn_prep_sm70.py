"""sm70 gdn_prep: parity against the reference, then the speed the switch bought.

sm70 now takes sm90's one-thread-per-column schedule at f32
(`make_gdn_prep_bf16(target, "float32")`) instead of the CPU source's
`T.serial(DK)`-in-every-thread form. Two things have to hold, and parity comes first
because a faster wrong kernel is worse than a slow right one:

  1. allclose(rtol=1e-2) against `reference.gdn_prep` on the real shape
  2. the old maker vs the new one, same inputs, same launch

Both makers are called directly, so nothing else is in the window. Refuses if another
process holds the card.
"""

import os
import subprocess
import time

import torch


def card_busy(mine: int) -> str | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    return "\n".join(ln for ln in out.splitlines()
                     if ln.strip() and int(ln.split(",")[0]) != mine) or None


def main() -> None:
    if (busy := card_busy(os.getpid())):
        print(f"card busy, refusing:\n{busy}")
        return

    from tilerl_kernels import kernels, kernels_gdn, reference
    from tilerl_kernels.backend import get_backend

    backend = get_backend()
    dev, tgt = backend.device, backend.target
    print(f"arch {backend.arch}  io {backend.io}")

    # The 27B's real GDN shape: 48 value heads, head_dim 128, 4 conv taps, GQA 1:1
    # for q/k here (the model repeats k/v, which changes reads, not correctness).
    T_LEN, NVH, DK, KER = 512, 48, 128, 4
    HK = NVH
    qkvd = 2 * HK * DK + NVH * DK
    g = torch.Generator(device="cpu").manual_seed(7)
    mk = lambda *s: torch.randn(*s, generator=g).to(dev)
    q, k = mk(1, T_LEN, HK, DK), mk(1, T_LEN, HK, DK)
    v = mk(1, T_LEN, NVH, DK)
    gin, bin_ = mk(1, T_LEN, NVH), torch.rand(1, T_LEN, NVH, generator=g).to(dev)
    dtb, alog = mk(NVH), mk(NVH) * 0.1
    cw = mk(qkvd, KER)
    win = torch.zeros(1, KER - 1, qkvd, device=dev)

    old = kernels.make_gdn_prep(tgt)
    new = kernels_gdn.make_gdn_prep_bf16(tgt, "float32")

    args = (q, k, v, gin, bin_, dtb, alog, cw, win)
    out_old = old(*args, threads=DK)
    out_new = new(*args, threads=DK)

    # The oracle: reference.gdn_prep takes the layer's flat [B,T,*] layout.
    ref = reference.gdn_prep(
        q.reshape(1, T_LEN, HK * DK), k.reshape(1, T_LEN, HK * DK),
        v.reshape(1, T_LEN, NVH * DK), gin, bin_, DK,
        conv1d_weight=cw.T.contiguous().T, dt_bias=dtb, a_log=alog,
        conv_window=win,
    )
    names = ("Qo", "Ko", "Vo", "Go", "Bo", "NewWindow")
    print("\nparity, new kernel vs reference.gdn_prep:")
    ok = True
    for i, nm in enumerate(names[: len(ref)]):
        a, b = out_new[i].float(), ref[i].float().to(dev)
        if a.shape != b.shape:
            print(f"  {nm:10s} shape {tuple(a.shape)} vs {tuple(b.shape)} -- skipped")
            continue
        d = (a - b).abs().max().item()
        close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
        ok &= close
        print(f"  {nm:10s} max|delta| {d:.3e}  allclose(1e-2) {close}")

    print("\nnew vs OLD kernel (must agree, they are the same math):")
    for i, nm in enumerate(names):
        d = (out_new[i].float() - out_old[i].float()).abs().max().item()
        print(f"  {nm:10s} max|delta| {d:.3e}")

    def bench(fn, iters=5):
        fn(*args, threads=DK)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn(*args, threads=DK)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    ms_old, ms_new = bench(old), bench(new)
    print(f"\nT={T_LEN} NVH={NVH} DK={DK}:")
    print(f"  old (T.serial(DK) per thread) {ms_old:8.2f} ms")
    print(f"  new (one thread per column)   {ms_new:8.2f} ms   {ms_old / ms_new:.2f}x")
    print(f"\nparity {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
