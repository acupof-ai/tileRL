"""Stream-quantize a bf16 HF checkpoint to tileRL NVFP4, one shard at a time.

Peak host RAM ≈ one shard (~4GB), so it fits the V100 box's 31GB. Reuses the
load_hf name mapping (_param_key_for) and fp4_param_keys so the output is
exactly what load_hf(cfg, fp4=True) reads back through its ``.wq`` branch —
no re-quantization on load. Non-fp4 tensors (norms, conv1d, embed, dt_bias,
a_log) pass through bf16 unchanged. lm_head is quantized when untied.

Usage:
  python scripts/quantize_nvfp4.py <src_bf16_dir> <dst_dir> <cfg_fn>
    cfg_fn = qwen38_27b | qwen36_27b
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch
from safetensors.torch import load_file, save_file

from tilerl import config as cfgmod
from tilerl.model import _param_key_for, fp4_param_keys
from tilerl_kernels.reference import pack_fp4, renorm_fp4_scale


def _pack_chunked(w, rows=2048):
    """pack_fp4 by row chunks: its argmin materializes [N,K//B,B,8] f32 at
    once (40 GB for lm_head 248320x5120), so slice N to bound peak memory."""
    w = w.to(torch.bfloat16)
    n = w.shape[0]
    if n <= rows:
        return pack_fp4(w)
    wqs, scs = [], []
    for i in range(0, n, rows):
        wq, sc = pack_fp4(w[i : i + rows])
        wqs.append(wq)
        scs.append(sc)
    return torch.cat(wqs, 0), torch.cat(scs, 0)


def main(src, dst, cfg_fn):
    cfg = getattr(cfgmod, cfg_fn)()
    fp4_keys = fp4_param_keys(cfg)
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("model-*.safetensors")) or [src / "model.safetensors"]
    assert shards and shards[0].exists(), f"no shards in {src}"

    index = {"metadata": {}, "weight_map": {}}
    n_q = n_pass = 0
    for si, shard in enumerate(shards):
        tensors = load_file(str(shard))
        out = {}
        for hf_name, t in tensors.items():
            key = _param_key_for(hf_name)
            if key is not None and key in fp4_keys and hf_name.endswith(".weight"):
                stem = hf_name.removesuffix(".weight")
                wq, scale = _pack_chunked(t)
                scale, oscale = renorm_fp4_scale(scale)
                out[stem + ".wq"] = wq.contiguous()
                out[stem + ".scale"] = scale.contiguous()
                out[stem + ".oscale"] = oscale.contiguous()
                n_q += 1
            else:
                out[hf_name] = t  # norms/conv/embed/dt_bias/a_log stay bf16
                n_pass += 1
        name = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(out, str(dst / name))
        for k in out:
            index["weight_map"][k] = name
        print(f"shard {si + 1}/{len(shards)}: {len(out)} tensors -> {name}", flush=True)
        del tensors, out

    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    # config.json: copy source, stamp the fp4 quant marker load_hf reads via cfg.
    for aux in ("config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.jinja"):
        s = src / aux
        if s.exists():
            (dst / aux).write_bytes(s.read_bytes())
    print(f"DONE: {n_q} linears quantized to NVFP4, {n_pass} tensors passed through bf16")


if __name__ == "__main__":
    main(*sys.argv[1:4])
