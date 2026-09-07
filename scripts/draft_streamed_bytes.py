"""What does ONE draft forward stream? The last operand of the 32-42% bracket.

`errors/2026-09-06-a-spec-rate-over-a-dense-roofline.md` corrects four sites that divided a
speculative tok/s by a dense single-token roofline. The identity needs no bytes:

    % of the speculative ceiling = (% of the dense roofline) / (tok per forward)

but the *share* it lands on does. The entry brackets it at **32-42%** and names the missing
operand: 32% assumes a draft forward streams ZERO bytes; 42% charges 3 x (0.85 + 0.80) GB,
where **0.85 is the head's RESIDENT size on disk, not what a forward multiplies by**. That is
the same defect `errors/2026-09-02-roofline-is-the-streamed-subset.md` fixed for the trunk --
`embed_tokens` is in the file and is gathered, not streamed -- applied to the draft.

So bucket the draft the way `check_scale_f16.py` buckets the trunk: per tensor, by role, at the
SERVED dtype, keeping only what a forward reads.

Three facts from the code decide what counts, and each is asserted rather than assumed:

1. `spec.read_head_params` (spec.py:530) SKIPS the head's own `lm_head`, `embed_tokens` and
   `final_norm` -- "the trunk's are shared". So a draft forward's readout bytes are the
   TRUNK's lm_head, and an embedding row is gathered, not streamed.
2. `engine._quantize_draft` (engine.py:72) re-serves every [N,K] with both dims >= 128 as fp4
   on sm70 (`not backend.has_kernel("linear_fp8")`): 4 bits per weight plus one f32 scale per
   32. So the served size is ~0.14x the bf16 checkpoint size, NOT the file's own bytes.
3. `DraftHead.width` = depth+1, and a depth-3 tick runs **3** draft forwards (spec.py:290).

Counts what a forward multiplies by; excludes the KV plane, which is a per-token read that
scales with context and is not part of a weight roofline.

  scripts/v100.sh 'cd ~/models/Qwen3.8-27B-NVFP4 && /usr/bin/python3 \
      $HOME/tilerl-v100/scripts/draft_streamed_bytes.py \
      --draft model-00018-of-00018.safetensors'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from safetensors import safe_open

#: What `_quantize_draft` re-serves a [N,K] projection as on sm70: 4-bit nibbles plus one f32
#: scale per 32 weights. Asserted against reference.pack_fp4's own output below.
_FP4_BITS = 4
_SCALE_BLOCK = 32

#: `_quantize_draft` only packs both-dims >= 128; anything smaller stays at its own dtype.
_QUANT_MIN = 128


def _served_bytes(t: torch.Tensor, quantized: bool, scale_bytes: int = 4) -> tuple[int, str]:
    """Bytes one forward reads for this tensor, and the kind it was counted as."""
    n = t.numel()
    if quantized:
        return n * _FP4_BITS // 8 + (n // _SCALE_BLOCK) * scale_bytes, f"fp4+s{scale_bytes}"
    return n * t.element_size(), str(t.dtype).removeprefix("torch.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="the head's safetensors shard")
    ap.add_argument("--trunk-lm-head-gb", type=float, default=0.80,
                    help="the trunk lm_head the head reads out through, from the 09-02 buckets")
    ap.add_argument("--depth", type=int, default=3)
    # 4, not `Backend.scale_io`'s 2: the two sites this entry corrects are 09-01, a day
    # before sm70's scale plane went f16 (backend.py:356, f709df8). A post-09-02 draft
    # rate must pass 2 -- carrying 4 there overstates a forward by 0.027 GB.
    ap.add_argument("--scale-bytes", type=int, default=4, choices=(2, 4),
                    help="served block-scale width: 4 = the f32 era these sites were measured in")
    ap.add_argument("--bandwidth", type=float, default=900.0, help="GB/s")
    ap.add_argument("--dense-gb", type=float, default=16.04,
                    help="trunk + lm_head per dense token, from the 09-02 buckets")
    args = ap.parse_args()

    #: (label, published tok/s, measured tok/forward) — the two sites the entry corrects.
    #: Both are depth 3; the tok/fwd is what each run measured, not d+1.
    args.sites = [
        ("wins/2026-09-01-…-gemv-packed-x-f16.md:83", 52.7, 2.95),
        ("wins/2026-09-01-…-attention-thread-redundancy:93", 46.5, 2.90),
    ]

    # The fp4 byte formula is a hypothesis until pack_fp4's own output confirms it: a wrong
    # block size or an extra oscale plane would move every number below. Pinned at 4 whatever
    # --scale-bytes says: pack_fp4 emits an f32 plane and materialize narrows it on the device
    # move, so comparing a narrowed model against pack_fp4's own bytes would fail for the one
    # reason that is not a modelling error.
    from tilerl_kernels import reference

    from tilerl.spec import _DRAFT_TOP, read_head_params
    probe = torch.randn(256, 512)
    wq, sc = reference.pack_fp4(probe)
    sc, osc = reference.renorm_fp4_scale(sc)
    real = sum(t.numel() * t.element_size() for t in (wq, sc, osc))
    model, _ = _served_bytes(probe, quantized=True, scale_bytes=4)
    assert abs(real - model) <= real * 0.02, (
        f"fp4 byte model is wrong: pack_fp4 emits {real} bytes for {tuple(probe.shape)} "
        f"(wq {wq.numel() * wq.element_size()}, scale {sc.numel() * sc.element_size()}, "
        f"oscale {osc.numel() * osc.element_size()}), the model says {model}"
    )
    print(f"fp4 byte model checked against pack_fp4: {model} vs {real} bytes on a 256x512")

    kept = read_head_params(args.draft, _DRAFT_TOP)

    # Every tensor in the file, so "kept" is a measured subset and not a claim.
    with safe_open(args.draft, "pt", device="cpu") as f:
        all_names = list(f.keys())
        file_bytes = 0
        for n in all_names:
            t = f.get_tensor(n)
            file_bytes += t.numel() * t.element_size()

    rows, streamed, resident = [], 0, 0
    for k in sorted(kept):
        t = kept[k]
        quant = t.ndim == 2 and t.shape[0] >= _QUANT_MIN and t.shape[1] >= _QUANT_MIN
        b, kind = _served_bytes(t, quant, args.scale_bytes)
        rows.append((k, tuple(t.shape), kind, b))
        resident += t.numel() * t.element_size()
        streamed += b

    print(f"draft shard: {args.draft}")
    print(f"  {len(all_names)} tensors in the file, {file_bytes / 1e9:.3f} GB as stored")
    print(f"  {len(kept)} kept by read_head_params (the rest are the trunk's, skipped)\n")
    print(f"{'tensor':>34} {'shape':>16} {'served as':>10} {'MB':>8}")
    for k, shape, kind, b in rows:
        print(f"{k:>34} {str(shape):>16} {kind:>10} {b / 1e6:>8.2f}")

    print(f"\n{'head, as stored (bf16)':>34} {resident / 1e9:>8.3f} GB")
    print(f"{'head, as SERVED (fp4 on sm70)':>34} {streamed / 1e9:>8.3f} GB")
    print(f"{'trunk lm_head it reads out via':>34} {args.trunk_lm_head_gb:>8.3f} GB")
    per_fwd = streamed / 1e9 + args.trunk_lm_head_gb
    print(f"{'ONE draft forward streams':>34} {per_fwd:>8.3f} GB")

    d = args.depth
    total = args.dense_gb + d * per_fwd
    fwd_sets = args.bandwidth / total
    print(f"\ndepth {d}: 1 trunk forward + {d} draft forwards")
    print(f"  bytes per tick        {args.dense_gb:.2f} + {d} x {per_fwd:.3f} = {total:.2f} GB")
    print(f"  dense roofline        {args.bandwidth / args.dense_gb:.1f} tok/s "
          f"({args.dense_gb:.2f} GB)")
    print(f"  tick-sets per second  {fwd_sets:.2f}")
    print(f"  perfect acceptance    {fwd_sets * (d + 1):.1f} tok/s at {d + 1} tok/set")

    # The bracket's own identity: a speculative ceiling is (sets/s) x (MEASURED tok per
    # forward-set), so the published rate's share uses the acceptance each site measured --
    # d+1 above is the perfect-acceptance variant and is NOT what 32-42% was computed against.
    print(f"\n{'site':>44} {'tok/s':>7} {'tok/fwd':>8} {'ceiling':>9} {'share':>7}")
    for label, rate, tpf in args.sites:
        ceil = fwd_sets * tpf
        print(f"{label:>44} {rate:>7.1f} {tpf:>8.2f} {ceil:>9.1f} {rate / ceil * 100:>6.1f}%")

    print("\nThe bracket's two ends, same identity, for the 52.7 site:")
    for label, gb in (("draft streams 0 bytes", 0.0),
                      ("draft charged 1.65 GB (the RESIDENT size)", 1.65),
                      ("draft charged the MEASURED bytes", per_fwd)):
        c = args.bandwidth / (args.dense_gb + d * gb) * 2.95
        print(f"  {label:>42} -> {c:>6.1f} tok/s, 52.7 is {52.7 / c * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
