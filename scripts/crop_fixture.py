"""Crop the local Qwen3.5-0.8B MLX-4bit into a 2-layer, vocab-1024 fixture.

One-shot reproducibility tool: the output lives in tests/fixtures/ and is
committed, so the real-weight test is hermetic (no external model on CI).
Keeps MLX affine-4bit format on disk so the loader's dequant path is exercised.

Usage: uv run python scripts/crop_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC = Path("/Users/bytedance/code/agent-infer/models/Qwen3.5-0.8B-MLX-4bit")
DST = Path(__file__).parent.parent / "tests" / "fixtures" / "qwen35-2layer-mlx4"
KEEP_LAYERS = {"0": "0", "3": "1"}  # src layer -> dst layer (GDN, then full-attn)
VOCAB = 1024


def main() -> None:
    tensors = load_file(str(SRC / "model.safetensors"))
    out: dict[str, object] = {}
    for name, t in tensors.items():
        if not name.startswith("language_model."):
            continue  # vision tower, etc.
        if name.startswith("language_model.model.embed_tokens"):
            out[name] = t[:VOCAB].contiguous()
            continue
        if name.startswith("language_model.model.norm"):
            out[name] = t
            continue
        parts = name.split(".")
        # language_model.model.layers.<i>....
        if len(parts) > 4 and parts[2] == "layers" and parts[3] in KEEP_LAYERS:
            parts[3] = KEEP_LAYERS[parts[3]]
            out[".".join(parts)] = t.contiguous()

    DST.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST / "model.safetensors"))

    cfg = json.loads((SRC / "config.json").read_text())
    text = cfg["text_config"]
    text["num_hidden_layers"] = len(KEEP_LAYERS)
    text["vocab_size"] = VOCAB
    text["layer_types"] = ["linear_attention", "full_attention"]
    (DST / "config.json").write_text(
        json.dumps({"text_config": text, "quantization": cfg["quantization"]})
    )
    mb = sum(t.storage().nbytes() for t in out.values()) / 1e6
    print(f"wrote {len(out)} tensors, {mb:.1f} MiB -> {DST}")


if __name__ == "__main__":
    main()
