# Pending-remote runbook — the gated work one card sitting completes

One sitting, in order. Every gate below shipped its CPU half with
`pending-remote` stamped on the card half; this is the exact list of commands
that turns them into owned numbers. The CPU-off-cuda output shape for each
command was run on 2026-09-11 and quoted below, so a card run whose shape
differs is a regression, not an environment quirk.

Run on the pod against the committed tree (`scripts/POD-VERIFY.md` covers
sync/commit discipline), with the checkpoint present at the tileRL-owned
`/work/tilerl-ckpt/Qwen3.8-27B-NVFP4` (the earlier shared `/work/Qwen3.8-27B-NVFP4` was
deleted by a cross-project dedup on 2026-09-11; see the 2026-09-11 checkpoint-deleted
error entry; the host copy is on
`iv-yeozpb5g5cbw80bls64e`, container `sglang-test`, 2026-09-11 — name the host
and `ls -l` before calling it absent).

**Card grant:** tileRL cards are **0, 1, 3, 6** once the aupai V4.1 block
dissolves. Per-card commands bind one physical card per run; multi-rank gates
pin `CUDA_VISIBLE_DEVICES` to two of them. Never assume 6/7 — that grant
expired.

**Marker count (grep the reviewer used):**
`grep -rn pending-remote src scripts tests docs --include='*.py' --include='*.md'`
reports **141 on main**; this runbook adds two prose mentions (the phrase in
this header and the step-3 column note), so the same grep reports **143 at this
file's head** — no new gated code marker. The executable gates are the seven
groups below; everything else the grep finds is dated wins/errors prose,
roadmap text, or the "also pending" tail.

---

## 1 — Calibrate every card (one per card, first — every later gate reads these)

```bash
for c in 0 1 3 6; do
  CUDA_VISIBLE_DEVICES=$c TILERL_TARGET=cuda uv run tilerl bench --calibrate --card 0
done
```

With one visible card the in-process index is always 0; the loop binds one
physical card per run.

- **Expected:** two lines per card —
  `appended hbm_bw_gbs        <v> GB/s <card name> card 0 -> <store path>` and
  `appended bf16_peak_tflops  <v> TFLOP/s … -> <store path>`. The line ends in
  the **store path** (`docs/experience/bench/measurements.jsonl`, or
  `$TILERL_BENCH_STORE`); the row id is read back from the file.
- **CPU output seen (refusal):**
  `error: --calibrate is cuda-only (large D2D copy + bf16 GEMM, CUDA events); run on the card: TILERL_TARGET=cuda tilerl bench --calibrate --card N. Off cuda the roofline prints pending-remote, never a datasheet number.`
- **Writes:** two rows per card, metrics `hbm_bw_gbs` (≥1 GiB D2D copy,
  read+write, CUDA-event median of 20) and `bf16_peak_tflops` (8192² bf16
  GEMM, 2n³ flops), `floor.kind=measured-best`, full 40-hex commit.

## 2 — Record steady-state residency and its static/transient split  ✅ on H20 card 2 (2026-09-11)

Shipped — 27B built-engine `--record-residency` on card 2, every derived row
equal measured to the byte: peak 76,451,655,680 = static 76,338,610,340 +
transient 113,045,340 (row `edb5d8328af3`), in
[wins/2026-09-11-h20-kernel-roofline-step3.md](../wins/2026-09-11-h20-kernel-roofline-step3.md).
That entry pins the 314-vs-315 cause: the captured decode graph reserves one
pad slot+block, so pool num_blocks is one more than usable_blocks.

```bash
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl serve \
  --model qwen38-27b --dry-run --record-residency
```

- **Expected:** the unified table then
  `appended device_resident_bytes peak <N> = static <N> + transient <N> (<card name>) -> <12-char id>`
  (this line ends in the row **id**, unlike step 1).
- **CPU output seen:** the same table renders (weights/state_slots/kv_pool +
  two budget rows header-only; 5 JSON rows); the append refuses:
  `error: --record-residency is cuda-only (device residency belongs to the card; the CPU tiny cell has no measured peak). Run on the card: TILERL_TARGET=cuda tilerl serve --dry-run --record-residency`
- **Writes:** one `device_resident_bytes` row,
  `shape={card,static,transient}` closing `peak = static + transient`,
  target sm90 with the physical card. The header-only variant
  (`--checkpoint DIR`) renders the table but writes nothing — residency needs a
  built engine.

## 3 — Kernel roofline with measured ms / bound / %bound  ✅ shuffled on H20 card 2 (2026-09-11)

Shipped — every timed row in (0,100] on sm90 card 2, decode B=1/B=8 and
prefill S=4096, against measured bw/bf16/**fp8** floors. See
[wins/2026-09-11-h20-kernel-roofline-step3.md](../wins/2026-09-11-h20-kernel-roofline-step3.md).
Two rules that run established: the ceiling is the kernel's MMA-dtype peak
(w4a8 prefill rides the fp8 peak, decode the bf16 one), and the timer times the
priced launch M (decode M=b, not b·s). Still pending: timing fixtures for the
fused attention/GDN/norm rows (ms `pending`; bounds already print) and a B=8
prefill column.

Timing resolves each row's kernel **by its weight face** (nvfp4 → linear_fp4,
fp8 → linear_fp8; fused attention/GDN/norms have no timing fixture).

```bash
CKPT=/work/tilerl-ckpt/Qwen3.8-27B-NVFP4  # tileRL-owned copy; see 2026-09-11 dedup error
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl bench --kernels \
  --checkpoint "$CKPT" --batches 1,8
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl bench --kernels \
  --checkpoint "$CKPT" --prefill 4096
```

- **CPU output seen (no calibration):** header
  `kernel count shape face bytes flops ms bound %bound`, then
  `# tiny decode tick B=1 s=4096, fp8 KV, nvfp4 weights (config face), floor device=tiny-cpu`
  and `# (no calibration row for this device: ms/bound/%bound pending-remote)`,
  every row's three measured columns `pending`.
- **With floors:** real `ms`/`bound` (`max(bytes/bw, flops/peak)`)/`%bound`; the
  `face` column is one of `nvfp4`, `fp8blk` (fp8 block-grid, no row scale),
  `fp8` (block grid + per-row scale), `bf16` — with `--checkpoint` the 27B
  shows all four populations (168/96/233); without it every GEMM is nvfp4.
  Decode ends in `TICK TOTAL` per batch, prefill in `PREFILL TOTAL`.
- **The `ms` column is eager per-launch, not the served tick.** Every one of
  the 497 timed kernels is dispatched separately, so each row carries a
  ~0.1 ms launch overhead its roofline `bound` does not. Measured in the engine
  loop with the CUDA decode graph captured (32 decode-only ticks, s=4096,
  card 6 2026-09-11): B=1 graph **11.68 ms** vs eager **47.99 ms** (4.11×,
  57% of the 6.6 ms bound); B=8 graph **25.98 ms** vs eager **64.69 ms**
  (2.49×, 30% of the 7.74 ms bound). Cite the engine-loop graph tick for
  served decode latency; use this table only to rank kernels by `ms − bound`
  (excess = launch overhead plus the kernel's own gap — at B=8 the excess is
  the M≤8 mma8 GEMV band, not dispatch). See
  `wins/2026-09-11-decode-tick-graph-vs-eager-roofline-gap.md`.
- **Writes:** nothing — a view over step 1/2 rows; copy the tables into the
  dated wins entry by hand.

## 4 — The 27B byte oracles (integer-exact; headers or live load_hf)  ✅ on H20 card 2 (2026-09-11)

Both env-gated exact-byte gates pass live on card 2 against
`/work/Qwen3.8-27B-NVFP4`: served-face bytes == header-derived == LIVE
`load_hf` tensor storage = **24,436,981,888 B over 1845 tensors**, including
both fp8 scale planes. `2 passed` (the two named gates run explicitly). One
adjacent test bug surfaced and is fixed in #510: the header dry-run block
expectation derived the decode-graph pad from a CPU RefBackend instead of the
built CUDA backend (305 vs 306). The gates remain integer-exact, no tolerance.

Run pytest from one invocation with both files as args (a `file -k` pair is not
valid pytest syntax):

```bash
export TILERL_27B_CKPT=/work/tilerl-ckpt/Qwen3.8-27B-NVFP4  # tileRL-owned copy; see 2026-09-11 dedup error
TILERL_TARGET=cuda uv run pytest tests/test_kernel_cost.py tests/test_memory_ledger.py \
  -k "faces or checkpoint or 27b" -q
# header-only dry-run, integer table against a free budget:
TILERL_TARGET=cuda uv run tilerl serve --model qwen38-27b --dry-run \
  --checkpoint "$TILERL_27B_CKPT" --device-free 60000000000
```

- **CPU output seen without the env:** `3 passed, 2 skipped` — the two skips are
  the 27B oracles (`set TILERL_27B_CKPT…`), everything else runs.
- **Expected on card:** `test_27b_served_weight_faces_equal_load_hf_resident_exact`
  and `test_27b_checkpoint_weights_row_matches_load_hf_resident_exact` pass when
  the served-face sum equals both the recorded load_hf resident total and a
  LIVE load_hf model's tensor storage, to the integer: **24,436,981,888 B** over
  1845 tensors; the materialized-pool gates include both fp8 scale planes.
  No tolerance — a mismatch is a naming/repack regression.
- **Writes:** nothing; change the recorded constant only after re-deriving it.

## 5 — GDN context-parallel world2 gates on the CUDA cells

Green on CPU gloo today; the card run drives the same tape through CUDA
collectives. Unique per-gate gloo ports let them run back to back; never run two
instances of the SAME gate concurrently.

```bash
export TILERL_TARGET=cuda
for g in gdn_world2 gdn_cp_gradcheck_world2 gdn_cp_tape_world2 gdn_halo_world2 cp_world2; do
  CUDA_VISIBLE_DEVICES=0,1 python3 tests/$g.py
done
CUDA_VISIBLE_DEVICES=0,1 python3 tests/gdn_halo_world2.py --no-halo   # control, must FAIL
```

- **Expected pass lines (seen on CPU 2026-09-11):**
  `gdn cp=2 zigzag: every chunk from its scanned prefix matches the sequential scan`,
  `affine scan reverse matches central differences`,
  `gdn_cp world2 tape matches global central differences worst ~4e-4`
  (global-loss central differences, floats over a plain Queue),
  `gdn halo cp=2: every chunk with its left context matches the sequential run`.
- **Control:** `--no-halo` exits 0 with `no-halo control: correctly FAILED`
  (the control is *supposed* to fail the rel check — that inverted exit code is
  the gate; a vacuous gate exits 1).
- The future P6 exit (8 cards, 32K gradient match to 1e-3, 256K fwd+bwd) is a
  separate run — this batch's CP gates are world 2. **Writes:** nothing.

## 6 — P1 judge recipe, two seeds, pre-registered acceptance

```bash
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl train \
    --recipe grpo-gsm8k-27b \
    --data /work/p1_gsm8k_train.jsonl \
    --eval-gsm8k /work/gsm8k_test.jsonl --eval-n 500 \
    --steps 100 --seed $s
done
```

Recipe-fixed: group 8, 256-token rollout, LoRA rank 16, self-judge on, cap 512.
Off cuda the command refuses at model load (the 27B needs its checkpoint dir).

- **Acceptance, BOTH seeds, pre-registered (roadmap P1):** GSM8K held-out (500)
  after − before ≥ **+5 pt** under the paired McNemar the per-question rows
  already write (SE 1.00–1.41 pt at 5–10% discordant; the unpaired MDE is
  7.70 pt and cannot see +5); MMLU (1000, add `--eval-mmlu 1000`)
  after ≥ before − **2 pt**; tied-group fraction < **50%**. Flat /
  non-collapse is the bar; +5/500 is the verdict. Then self-OPD under the same
  gate.
- **Writes:** `runs/<id>/manifest.json` per seed — inputs with data hashes,
  metrics, gates, and the engine block including the memory rows (#475) — plus
  eval rows in `measurements.jsonl`.

## 7 — Recapture-after-update: token equality then wall clock (P2.0 step 0)

The card gate does not exist yet; this section is the spec. Its CPU half
(same-seed captured vs eager token equality after an update step) is built and
green before the pod run.

- **Token equality:** same seeds, one in-place optimizer step, rollouts under
  the kept graph with the f32 cast refilled produce the **same tokens** as eager
  decode. Drift means the refill or an in-place assumption in AdamW/Adafactor/ISO
  `step_one` broke.
- **Card wall clock:** captured RL rollout tok/s at group 8 within **5%** of
  plain captured decode (the draft costs 4.9× eager today); result appended as
  that step's bench row.

---

## Also pending-remote in the grep (pre-existing, not created by this batch)

- `tests/test_ops_parity.py` — fp8 sm90 kernel parity (C backend has no fp8).
- `tests/test_fp4_scale_e4m3_parity.py` — e4m3 scale path on sm90.
- `tests/test_rmsnorm_f32_tape.py` — skipped sm90 bring-up arms (no kernels yet).
- `scripts/bench_api_routes.py` — request-overhead real number on the V100.
- `src/tilerl/recipes.py` — four recipe status strings flip as the P1/P3 runs
  above land.
