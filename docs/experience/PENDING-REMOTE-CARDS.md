# Pending-remote runbook — the gated work one card sitting completes

One sitting, in order. Every gate below shipped its CPU half with
`pending-remote` stamped on the card half; this is the exact list of commands
that turns them into owned numbers. Run on the pod against the committed tree
(`scripts/POD-VERIFY.md` covers sync/commit discipline), with
`/work/Qwen3.8-27B-NVFP4` present (it was found on host
`iv-yeozpb5g5cbw80bls64e`, container `sglang-test`, 2026-09-11 — name the host
and `ls -l` before calling it absent).

**Card grant:** tileRL cards are **0, 1, 3, 6** once the aupai V4.1 block
dissolves. Per-card commands take `--card N`; multi-rank gates pin
`CUDA_VISIBLE_DEVICES` to two of them. Never assume 6/7 — that grant expired.

Completeness: `grep -rn pending-remote src scripts tests docs` reports 138
markers, but most are immutable prose in dated wins/errors and roadmap text.
The **executable** gates are the seven groups below; the "also pending" tail
lists the remaining code-side skips the grep finds, none of which this batch
created.

---

## 1 — Calibrate every card (one per card, first — every later gate reads these)

```bash
for c in 0 1 3 6; do
  CUDA_VISIBLE_DEVICES=$c TILERL_TARGET=cuda uv run tilerl bench --calibrate --card 0
done
```

(`--card` is the device index inside the process, so with one visible card it
is always 0; the loop binds one physical card per run.)

- **Expected:** two lines per card, `appended hbm_bw_gbs … GB/s` and
  `appended bf16_peak_tflops … TFLOP/s`, each ending `-> <12-char id>`.
- **Writes:** two rows in `docs/experience/bench/measurements.jsonl`, metrics
  `hbm_bw_gbs` (≥1 GiB D2D copy, read+write, CUDA-event median of 20) and
  `bf16_peak_tflops` (8192² bf16 GEMM, 2n³ flops), `floor.kind=measured-best`,
  full 40-hex commit. A datasheet number is refused; off cuda the command exits
  with the on-card invocation printed.

## 2 — Record steady-state residency and its static/transient split

```bash
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl serve \
  --model qwen38-27b --dry-run --record-residency
```

- **Expected:** the unified memory table (weights/state_slots/kv_pool/transient
  with derived/measured/delta + `device_total`) then
  `appended device_resident_bytes peak <N> = static <N> + transient <N> (<card
  name>) -> <id>`.
- **Writes:** one `device_resident_bytes` row in `measurements.jsonl`,
  `shape={card,static,transient}` closing `peak = static + transient`,
  target sm90 with the physical card. Off cuda this refuses by design (no
  card-less sm90 row). The header-only variant
  (`--checkpoint /work/Qwen3.8-27B-NVFP4`) renders the same table but writes
  nothing — residency needs a built engine.

## 3 — Kernel roofline with measured ms / bound / %bound

The table renders pending-remote until step 1 rows exist for that exact device
name. Timing resolves each row's kernel **by its weight face** (nvfp4 →
linear_fp4, fp8 → linear_fp8).

```bash
CKPT=/work/Qwen3.8-27B-NVFP4
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl bench --kernels \
  --checkpoint "$CKPT" --batches 1,8
CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl bench --kernels \
  --checkpoint "$CKPT" --prefill 4096
```

- **Expected:** one row per kernel with `face` (nvfp4/fp8blk/bf16), declared
  bytes/flops, measured `ms`, the roofline `bound`
  (`max(bytes/bw, flops/peak)` against step 1's floors), and `%bound`;
  fused attention/GDN/norm rows stay `pending` in `ms` (no engine-shaped
  fixture). Header line names the floor device; prefill ends in
  `PREFILL TOTAL`, decode in one `TICK TOTAL` per batch.
- **Writes:** nothing — this is a view over the step 1/2 rows; record the
  tables into the dated wins entry by hand.

## 4 — The 27B byte oracles (integer-exact, headers or live)

```bash
export TILERL_27B_CKPT=/work/Qwen3.8-27B-NVFP4
TILERL_TARGET=cuda uv run pytest tests/test_kernel_cost.py -k 27b
TILERL_TARGET=cuda uv run pytest tests/test_memory_ledger.py -k checkpoint_or_faces -q
TILERL_TARGET=cuda uv run tilerl serve --model qwen38-27b --dry-run \
  --checkpoint "$TILERL_27B_CKPT" --device-free 60000000000
```

- **Expected:** `test_27b_served_weight_faces_equal_load_hf_resident_exact`
  passes when the served-face sum equals both the recorded load_hf resident
  total and a **live** load_hf model's tensor storage, to the integer:
  **24,436,981,888 B** over 1845 tensors; the memory-ledger face tests assert
  the same against materialized pools (fp8 scale planes included); the dry-run
  prints weights + fitted blocks from headers alone. A mismatch is a naming or
  repack regression, not a tolerance issue — there is no band.
- **Writes:** nothing; paste any new integer into the test's recorded constant
  only after re-deriving it.

## 5 — GDN context-parallel world2 gates on the CUDA cells

These are green on CPU gloo today; the card run confirms the same tape through
CUDA collectives (kernels 1 and 4 this batch shipped).

```bash
export TILERL_TARGET=cuda
for g in gdn_world2 gdn_cp_gradcheck_world2 gdn_cp_tape_world2 gdn_halo_world2 cp_world2; do
  CUDA_VISIBLE_DEVICES=0,1 python3 tests/$g.py
done
# red control for the halo adjoint must fail LOUD:
CUDA_VISIBLE_DEVICES=0,1 python3 tests/gdn_halo_world2.py --no-halo
```

- **Expected:** each ends on its explicit pass line, e.g.
  `gdn_cp world2 tape matches global central differences`, with per-location
  RMS-rel ~1e-3 or tighter (the tape gates use global-loss central differences,
  floats over a plain Queue); the halo gate's kernel-1 arm stays exact-only;
  `--no-halo` is the control and must exit non-zero (routed tails zeroed).
  Unique per-gate gloo ports mean these can run back to back, but two
  instances of the SAME gate cannot share a port.
- **Writes:** nothing. The larger P6 exit (8 cards, 32K gradient match to 1e-3,
  256K fwd+bwd) is a separate, future run — this batch's gates are world 2.

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

(The recipe fixes group 8, 256-token rollout, LoRA rank 16, self-judge on,
cap 512; MMLU is scored when `--eval-mmlu N` is added.)

- **Acceptance, BOTH seeds, decided before the run (roadmap P1):** GSM8K
  held-out (500) after − before ≥ **+5 pt** under the paired McNemar the
  per-question rows already write (SE 1.00–1.41 pt at 5–10% discordant; the
  unpaired comparison's 7.70 pt MDE cannot see +5); MMLU (1000)
  after ≥ before − **2 pt**; tied-group fraction < **50%** (else the task is
  saturated — move to MATH). Flat / non-collapse is the bar; +5/500 is the
  verdict. Then self-OPD under the same gate.
- **Writes:** `runs/<id>/manifest.json` per seed with inputs (data hashes),
  metrics, pass/fail gates, and the engine block including step 2's memory
  rows; eval rows land in `measurements.jsonl`.

## 7 — Recapture-after-update: token equality then wall clock (P2.0 step 0)

The gate does not exist yet — this section is the spec its CPU half satisfies
before the card run.

- **Token equality (build on tiny, CPU-gateable now):** same seeds, an update
  step, then rollouts under the kept-and-refilled captured graph produce the
  **same tokens** as eager decode. This is the mechanism gate; distribution
  drift means the f32-cast refill (or an in-place assumption in
  AdamW/Adafactor/ISO `step_one`) broke.
- **Card wall clock:** captured RL rollout tok/s at group 8 within **5%** of
  plain captured decode (today the draft costs 4.9× eager). Command shape when
  the gate lands:
  ```bash
  CUDA_VISIBLE_DEVICES=0 TILERL_TARGET=cuda uv run tilerl bench [the captured-rollout arm]
  ```
  measured against the same shape's captured-decode baseline, result appended
  as the step's bench row.

---

## Also pending-remote in the grep (not created by this batch)

- `tests/test_ops_parity.py` — fp8 sm90 kernel parity (the C backend has no fp8
  type).
- `tests/test_fp4_scale_e4m3_parity.py` — e4m3 scale path on sm90.
- `tests/test_rmsnorm_f32_tape.py` — sm90 bring-up arms currently skipped (no
  kernels to check yet).
- `scripts/bench_api_routes.py` — request-overhead real number on the V100.
- `src/tilerl/recipes.py` — four recipe `status` strings flip from
  pending-remote as the P1/P3 runs above land.
