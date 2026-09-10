"""The training plan's derived rows equal the bytes the trainer allocates.

Same single-formula discipline as the serving ledger (test_memory_ledger.py):
adapter/optimizer/frame rows are priced from the precision roles over
param_specs, and the tape row equals the STORAGE of the RecordingBackend
entries after one recorded forward — counted on the real tiny tape, not guessed
from a formula. A mutant that drops one role or changes the head entry set must
go red.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tilerl.autograd import Adafactor, AdamW, RecordingBackend, Tape
from tilerl.cli import _build_model
from tilerl.config import tiny
from tilerl.iso import ISO
from tilerl.memory import (
    adafactor_state_row,
    adamw_state_row,
    adapter_row,
    iso_frame_row,
    tape_row,
    train_plan,
)
from tilerl.model import add_lora, drop_quantized, param_specs
from tilerl.testing import RefBackend
from tilerl.train import _training_kv


def _adapter_bytes(trainable: dict) -> int:
    return sum(t.numel() * t.element_size() for t in trainable.values())


def _adamw_bytes(opt: AdamW) -> int:
    return sum(t.numel() * t.element_size() for t in (*opt._m.values(), *opt._v.values()))


def _real_tape_bytes(model, backend, ids, positions, kv, segment) -> int:
    tape = Tape()
    with torch.no_grad(), tape:
        model.forward(ids, positions, kv, RecordingBackend(backend), segment=segment)
    return sum(e.output.numel() * e.output.element_size() for e in tape._entries)


def _ids(cfg, b, s):
    return (np.arange(b * s) % max(cfg.vocab_size - 1, 1)).astype(np.int64).reshape(b, s)


def test_adapter_and_adamw_rows_equal_lora_trainer_storage():
    cfg, model = _build_model("tiny", seed=0, keep_master=False)
    rank = 4
    trainable = add_lora(model, rank=rank)
    spec = param_specs(cfg)
    assert adapter_row(spec, rank).n == _adapter_bytes(trainable)

    opt = AdamW()
    from tilerl.train import rl_step
    ids = _ids(cfg, 2, 12)
    rl_step(model, ids, np.ones(2), np.ones(2, dtype=np.int64), RefBackend(), opt,
            trainable=trainable)
    assert adamw_state_row(spec, rank, adapter=True).n == _adamw_bytes(opt)


def test_tape_row_equals_the_real_recorded_tape_layer_segments():
    """B*S > 1280 takes segment='layer' (train._MLP_SEGMENT_MAX_T): the tape keeps
    only the embedding, the per-layer boundary hiddens, the final norm, the head
    tensors and the logits — every in-layer activation is recomputed in backward."""
    cfg, model = _build_model("tiny", seed=0, keep_master=False)
    add_lora(model, rank=4)
    backend = RefBackend()
    b, s = 1, 1300
    ids = _ids(cfg, b, s)
    kv = _training_kv(model, b, s, device=backend.device)
    tapeb = _real_tape_bytes(model, backend, ids, np.arange(s, dtype=np.int64), kv, "layer")
    row = tape_row(cfg, b, s, lora_rank=4)
    assert row.n == tapeb, f"plan {row.n} != real tape {tapeb}"


def test_tape_row_full_sft_layer_segments():
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    backend = RefBackend()
    b, s = 1, 1300
    ids = _ids(cfg, b, s)
    kv = _training_kv(model, b, s, device=backend.device)
    tapeb = _real_tape_bytes(model, backend, ids, np.arange(s, dtype=np.int64), kv, "layer")
    row = tape_row(cfg, b, s, lora_rank=None)
    assert row.n == tapeb


def test_full_sft_adafactor_and_iso_rows_equal_allocator_storage():
    cfg, model = _build_model("tiny", seed=0, keep_master=True)
    drop_quantized(model)
    spec = param_specs(cfg)
    ada = Adafactor()
    iso = ISO(ada)
    iso.begin()
    for p in model.params.values():
        iso.step_one(p, torch.zeros_like(p))
    ada_b = sum(x.numel() * x.element_size()
                for st in ada._state.values() for x in st)
    # frames exist only for 2-D params, one (U, S, V) triple each
    frame_b = sum(u.numel() * u.element_size() + ss.numel() * ss.element_size()
                  + v.numel() * v.element_size() for u, ss, v in iso._frames.values())
    assert adafactor_state_row(spec, iso=True).n == ada_b
    assert iso_frame_row(spec).n == frame_b


def test_train_plan_holds_every_role_and_a_dropped_role_goes_red():
    cfg = tiny()
    rows = {(r.owner, r.tier): r.n for r in train_plan(cfg, 1, 1300, lora_rank=4)}
    assert set(rows) == {("adapter", "device"), ("optimizer_state", "device"),
                         ("tape", "device")}
    assert all(n > 0 for n in rows.values())

    iso_rows = {(r.owner, r.tier): r.n for r in train_plan(cfg, 1, 1300, optim="iso")}
    assert set(iso_rows) == {("optimizer_state", "device"), ("frame", "host"),
                             ("tape", "device")}

    # mutant: dropping one role removes its bytes, and every role row is positive
    # over the exact storage it names (so a zero/absent row cannot masquerade).
    assert ("frame", "host") not in rows and iso_rows[("frame", "host")] > 0


def test_train_dry_run_cli_prints_the_plan_rows_header_only(tmp_path, capsys):
    """`train --dry-run --recipe` builds nothing and prints the same tier/owner
    table as serve; B/S come from the micro-batch and the sequence length."""
    import json

    from tilerl import cli

    argv = ["train", "--model", "tiny", "--rl", "--group", "8", "--micro", "2",
            "--max-new-tokens", "1300", "--lora-rank", "4", "--dry-run", "--json"]
    cli.cmd_train(cli._build_parser().parse_args(argv))
    got = {(r["tier"], r["owner"]): r["derived"] for r in json.loads(capsys.readouterr().out)}
    want = {(r.tier, r.owner): r.n for r in train_plan(tiny(), 2, 1300, lora_rank=4)}
    assert got == want  # CLI prints exactly the storage-exact plan rows
    assert got[("device", "adapter")] == adapter_row(param_specs(tiny()), 4).n


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
