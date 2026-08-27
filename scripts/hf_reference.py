"""External ground truth: run the checkpoint through HF transformers and dump
per-layer hidden-state norms + the greedy first token, so tileRL's collapse
(check 2) can be bisected against a reference that is known-correct. Every
tileRL op passes parity at real dims and attention/GDN match the HF source by
inspection, so the bug is wiring or weight interpretation — only an external
forward can localize it.

  python3 -u scripts/hf_reference.py /data00/Qwen3.8-27B-NVFP4 --gpu 6

Prints layer 0/1/2/3/last hidden norms and the argmax token for
'The capital of France is'. Compare against tileRL's health_probe residuals.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--gpu", type=int, default=6)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.source)
    model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype="auto", trust_remote_code=True
    ).cuda()
    model.eval()

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    print(f"prompt ids: {ids.tolist()}")

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple: embedding, then one per layer
    print(f"\n{len(hs)} hidden states (embed + {len(hs)-1} layers)")
    print(f"{'layer':<8} {'norm(last tok)':>16}")
    print("-" * 26)
    for i, h in enumerate(hs):
        name = "embed" if i == 0 else f"L{i-1}"
        if i <= 4 or i >= len(hs) - 2:
            print(f"{name:<8} {h[0, -1].float().norm().item():>16.4e}")

    logits = out.logits[0, -1]
    top = torch.topk(logits.float(), 5)
    print(f"\nlogits norm {logits.float().norm().item():.4e}")
    print(f"argmax token: {top.indices[0].item()} = {tok.decode([top.indices[0].item()])!r}")
    print(f"top5 ids: {top.indices.tolist()}")
    print(f"top5 decoded: {[tok.decode([t]) for t in top.indices.tolist()]}")

    # Dump every layer's last-token hidden vector + the prompt ids + logits so
    # tileRL can bisect element-wise: the first layer whose vector diverges
    # (cosine << 1 or relerr >> quant tol) against tileRL's own is the bug site.
    torch.save(
        {
            "ids": ids.cpu(),
            "hidden": torch.stack([h[0, -1].float().cpu() for h in hs]),  # [65, 5120]
            "logits": logits.float().cpu(),
        },
        "/work/hf_ref.pt",
    )
    print("\nsaved /work/hf_ref.pt (ids, per-layer hidden [65,5120], logits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
