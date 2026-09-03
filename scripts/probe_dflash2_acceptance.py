"""Acceptance of one DFlash2 block draft against the trunk's own greedy run.

  CUDA_VISIBLE_DEVICES=2 python scripts/probe_dflash2_acceptance.py

Reproduces wins/2026-09-03-dflash2-block-drafter.md. The control arms are the
point: `+1 norm fold` applies the zero-centered fold the loader skips for this
format, and `tap 1 zeroed` removes the only path by which a later block slot
sees an earlier one. Without them the headline number says nothing.
"""

import os
import sys

import numpy as np
import torch
from tilerl_kernels.backend import Backend, resolve_target

from tilerl.config import qwen38_27b
from tilerl.model import load_hf
from tilerl.prompt import render_chat
from tilerl.spec import load_draft
from tilerl.tokenizer import get_tokenizer
from tilerl.train import _training_kv

SRC = os.environ.get("TILERL_QWEN38_SOURCE", "/work/Qwen3.8-27B-NVFP4")
DRAFT = os.environ.get("TILERL_DFLASH2_SOURCE", "/work/Qwen3.8-27B-DFlash2")
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Name the capital of France and one thing it is famous for.",
    "Explain in two sentences why the sky is blue.",
    "Sort the list [5, 3, 9, 1] in ascending order and show the result.",
]


def main() -> int:
    backend = Backend(resolve_target())
    model = load_hf(qwen38_27b(), SRC, fuse_projections=True)
    model.params = backend.materialize(model.params)
    head = load_draft(model, os.path.join(DRAFT, "model.safetensors"))
    head.params = backend.materialize(head.params)
    tok = get_tokenizer(SRC)
    pristine = {k: v.clone() for k, v in head.params.items()}
    block, groups, taps = head.dcfg.block_size, head.groups, tuple(head.dcfg.target_layers)
    print(f"{backend.target} {head.dcfg}", flush=True)

    def forward(ids, aux_layers=()):
        kv = _training_kv(model, 1, len(ids), device=backend.device)
        out: list = []
        logits = model.forward(
            np.array([ids], dtype=np.int64),
            np.arange(len(ids)),
            kv,
            backend,
            hidden_out=out,
            aux_layers=aux_layers,
        )
        return logits, out

    # The trunk's own greedy run: the anchor it commits, then block-1 more.
    truths = []
    for text in PROMPTS:
        prompt = tok.encode(render_chat([("user", text)], thinking=False))
        ids = list(prompt)
        for _ in range(block):
            ids.append(int(forward(ids)[0][0, -1].argmax()))
        truths.append((prompt, ids[len(prompt)], ids[len(prompt) + 1 :]))

    def fold_norms():
        for k, v in head.params.items():
            if k.endswith(("norm", "hidden_norm", "q_norm", "k_norm")):
                head.params[k] = (v.float() + 1.0).to(v.dtype)

    def kill_second_tap():
        for k, v in head.params.items():
            if k.endswith(".base"):
                v[:, 1] = 0
            elif k.endswith("_conv.proj"):
                for side in range(2):
                    v[side * 2 * groups + groups : (side * 2 + 2) * groups] = 0

    for name, mutate in (
        ("shipped", None),
        ("+1 norm fold", fold_norms),
        ("tap 1 zeroed", kill_second_tap),
    ):
        head.params = {k: v.clone() for k, v in pristine.items()}
        if mutate:
            mutate()
        lens = []
        for prompt, anchor, truth in truths:
            aux = torch.cat(forward(prompt, aux_layers=taps)[1][:-1], dim=-1)
            drafted = head.draft(aux, np.arange(len(prompt)), anchor, backend)
            n = 0
            while n < len(truth) and drafted[n] == truth[n]:
                n += 1
            lens.append(1 + n)
        print(
            f"{name:14s} mean acceptance {sum(lens) / len(lens):.2f} of {block}  {lens}", flush=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
