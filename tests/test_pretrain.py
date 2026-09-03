"""Hermetic pretrain gate: JSONL packing, the pretrain loop (finite loss,
params move, checkpoints land), and a save_hf -> load_hf roundtrip whose
loaded model forwards identically."""

from __future__ import annotations

import json
import math
import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import numpy as np
import torch
from tilerl_kernels.backend import get_backend

from tilerl.autograd import AdamW
from tilerl.config import tiny
from tilerl.engine import SamplingParams, build_engine
from tilerl.model import build_random, load_hf, save_hf
from tilerl.server import ByteTokenizer
from tilerl.train import JsonlDataset, pretrain


def _write_jsonl(path, texts) -> None:
    path.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")


def test_jsonl_dataset_packs_and_pads(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, ["hello world", "second document"])
    ds = JsonlDataset(p, ByteTokenizer(), seq_len=8, eos_token_id=0)
    seqs = list(ds)
    assert seqs and all(s.shape == (8,) and s.dtype == np.int64 for s in seqs)
    # iteration is deterministic (file order; pretrain owns the shuffle)
    assert [s.tolist() for s in ds] == [s.tolist() for s in seqs]
    # a corpus shorter than seq_len yields one eos-padded sequence
    p2 = tmp_path / "short.jsonl"
    _write_jsonl(p2, ["abc"])
    short = list(JsonlDataset(p2, ByteTokenizer(), seq_len=64, eos_token_id=0))
    assert len(short) == 1 and short[0].shape == (64,)


def test_pretrain_clips_params_only(tmp_path, monkeypatch):
    """The GDN initial state is a tape leaf: tape.backward yields a grad for
    it, and clip_grad_norm must never see it (regression: 27 params -> 28
    grads, over-clip up to 6%/step)."""
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, ["the quick brown fox jumps over the lazy dog"])
    cfg = tiny()
    model = build_random(cfg, seed=42)
    backend = get_backend()
    import tilerl.train as train_mod

    seen: dict = {}
    real = train_mod.clip_grad_norm

    def spy(grads, max_norm):
        seen.update(grads)
        return real(grads, max_norm)

    monkeypatch.setattr(train_mod, "clip_grad_norm", spy)
    pretrain(
        model,
        JsonlDataset(p, ByteTokenizer(), seq_len=32, eos_token_id=0),
        backend,
        AdamW(lr=3e-3),
        steps=1,
        seed=0,
    )
    param_ids = {id(v) for v in model.params.values()}
    assert seen and set(seen) <= param_ids


def test_pretrain_step_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(
        p,
        [
            "the quick brown fox jumps over the lazy dog",
            "tileRL pretrains tiny language models from scratch",
            "packed sequences train faster than padded ones",
        ],
    )
    cfg = tiny()
    backend = get_backend()
    model = build_random(cfg, seed=42)
    before = {k: v.clone() for k, v in model.params.items()}
    optimizer = AdamW(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    dataset = JsonlDataset(p, ByteTokenizer(), seq_len=32, eos_token_id=0)

    losses = pretrain(model, dataset, backend, optimizer, steps=5, seed=0)

    assert len(losses) == 5
    assert all(math.isfinite(l) for l in losses)
    assert any(not torch.equal(model.params[k].cpu(), before[k]) for k in before)

    # periodic + final checkpoints land on disk
    runs = tmp_path / "runs"
    model2 = build_random(cfg, seed=42)
    pretrain(
        model2,
        dataset,
        backend,
        AdamW(lr=3e-3),
        steps=4,
        ckpt_dir=runs,
        ckpt_every=2,
        seed=0,
    )
    assert (runs / "step_2" / "model.safetensors").exists()
    assert (runs / "final" / "config.json").exists()

    # save_hf -> load_hf: the loaded model forwards identically
    ckpt = tmp_path / "ckpt"
    save_hf(model, ckpt)
    loaded = load_hf(cfg, str(ckpt))

    def generate(m) -> list[int]:
        engine = build_engine(
            cfg, m, backend, num_blocks=8, num_slots=4, max_batch=4, max_total_tokens=512
        )
        rid = engine.submit(
            list(range(1, 17)),
            SamplingParams(temperature=0.0, top_p=1.0, max_new_tokens=4, seed=0),
        )
        # poll() clears finished entries — accumulate, don't probe twice.
        done = {}
        for _ in range(64):
            engine.step()
            done.update(engine.poll())
            if rid in done:
                break
        engine.shutdown()
        return done[rid]

    assert generate(loaded) == generate(model)
