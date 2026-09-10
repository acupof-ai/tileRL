# The training engine has a plan too — adapter/optimizer/frame/tape rows, derived, CPU-only — 2026-09-11

> Status: derived (tiny rows byte-exact against the live trainer; 27B rows
> derived, pending-remote for an allocator measurement).

## Context

`memory.plan` priced the serving engine; the training engine was outside it
("tape outside plan until training shares it"). The roles that hold training
bytes already exist in `precision` — `adapter` (bf16), `optimizer_state`
(f32), `frame` (f32) — but nothing derived their totals, and the largest and
least-obvious row, the reverse tape, had no derivation at all.

## What worked

`memory.train_plan(cfg, b, s, *, lora_rank, optim)` returns the training rows
alongside the serving `plan`, using the same `nbytes` arithmetic over
`param_specs` — no second table:

- **LoRA GRPO/OPD** (`lora_rank` set): `adapter` = the bf16 A/B pairs
  `add_lora` actually attaches; `optimizer_state` = AdamW's two f32 moments
  over those pairs.
- **full SFT** (`optim` adafactor/iso): `optimizer_state` over the trained
  tensors; ISO adds a **host**-tier `frame` row (U, S, V f32 per 2-D weight,
  host-resident on CUDA). Under ISO the Adafactor factors live on the *frames*
  (U [N,r] and V [K,r]), not the weights — the row prices N+r and K+r per 2-D
  param, which is what the allocator holds.
- **tape** = what `RecordingBackend` retains after one recorded forward in
  `segment="layer"` (the mode `train._step` picks at B*S > 1280): the bf16
  embedding, one f32 boundary hidden per layer, the final-norm hidden, the head
  output — plus, with an adapter, its A delta and two more v-wide tensors
  (frozen-base output and residual add). Everything inside a layer is recomputed
  in backward, so it never stays on the tape.

`tilerl train --dry-run --recipe X [--batch B] [--train-seq-len S]` prints the
same tier/owner/bytes table as serve, building nothing; B defaults to
`--micro` then `--group`, S to `--max-new-tokens`.

27B derived rows (`train --dry-run`):

| recipe | adapter | optimizer_state | frame (host) | tape |
|---|---:|---:|---:|---:|
| grpo-gsm8k-27b, B=1 S=512 | 249.7 MiB | 998.7 MiB | — | 1,541.4 MiB |
| grpo-gsm8k-27b, B=1 S=1024 | 249.7 MiB | 998.7 MiB | — | 3,082.9 MiB |
| sft-iso-27b, B=1 S=1024 | — | 51.4 MiB | 146,521 MiB (~143 GiB) | 1,048.6 MiB |

The ISO frame total is the "fp32 frames of the 27B are ~200 GiB" named in
iso.py measured from param_specs (this count is per 2-D weight's U+S+V).

## Gates (`tests/test_train_plan.py`)

Every derived row is asserted against the bytes the trainer really allocates,
not a parallel formula:

- `test_adapter_and_adamw_rows_equal_lora_trainer_storage` — sum of the
  attached adapter tensors and AdamW `_m`/`_v` after a real `rl_step`.
- `test_tape_row_equals_the_real_recorded_tape_layer_segments` and
  `test_tape_row_full_sft_layer_segments` — plan row == Σ
  `output.numel()*element_size()` over the entries of a real RecordingBackend
  tape (LoRA and full SFT), at B×1300.
- `test_full_sft_adafactor_and_iso_rows_equal_allocator_storage` — a real
  ISO(Adafactor) step's state dict and frame dict.
- `test_train_plan_holds_every_role_and_a_dropped_role_goes_red` — the role
  set per mode; ISO's host frame appears only there.
- `test_train_dry_run_cli_prints_the_plan_rows_header_only` — the CLI prints
  exactly these rows, measured/delta null.

The tape gate is the load-bearing one: an initial formula that counted the
final logits twice, or priced the tied tiny head's adapter tensors as absent,
went red against the counted tape.

## Rule

Tape bytes are counted from the recording set after one forward, not guessed
from "activations are B*S*hidden": the layer segments discard everything but
the boundaries, and the head keeps an adapter-shaped entry set. An optimizer
row prices the tensors the optimizer actually keys state by — ISO keys
Adafactor on the SVD frames, not the weights.

## Results

| date | machine | target | rows |
|---|---|---|---|
| 2026-09-11 | local (tiny exact) | cpu | adapter / optimizer_state / [frame host] / tape |
| 2026-09-11 | derived 27B | cpu | grpo B=1/S=512 1.54 GiB tape; sft-iso 143 GiB host frames |

27B measured columns stay pending-remote until a training run runs on a card.
