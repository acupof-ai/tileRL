# H20 pod verification — native fp4 w4a8

Two lanes, and they are independent.

- **Kernel lane** — the fp4 decode GEMV's `(micro_size_k, GROUP)` grid
  (`scripts/bench_gemv_micro.py`). This is the day's actual decision. It needs
  no checkpoint, no model load and no valid config, so it is immune to every
  abort mode in the other lane.
- **Model lane** — `scripts/verify_h20_fp4.py`'s five checks on the 27B. Worth
  35 minutes, and it can die at second two on a config field.

Run the two cheap go/no-go probes, launch the model lane detached, then run the
kernel lane against its load window. Dev-only tooling — no bench entry for the
harness itself, but the numbers it prints are what the wins entry gets updated
with.

**Non-negotiable:** GPUs 0-5 are the user's own training run, 100% util and
~94 GiB each. Only **6 and 7** are ours. The script refuses to start on
anything else and refuses a GPU that is busy — do not talk it out of that.

---

## Before the pod (Mac, no GPU)

`pod_sync.sh` tars the working tree (`--exclude=.git`) while step 4's launch
line stamps `BENCH_COMMIT` from the Mac's HEAD — anything uncommitted produces
a report naming a commit that does not contain what ran. **Commit first**, then
confirm the gates at the committed tree:

```bash
TILERL_TARGET=cpu uv run pytest -q && TILERL_TARGET=metal uv run pytest -q
uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/
uv run python scripts/_sweep_gemv_micro.py     # codegen + index gate, no GPU
```

`_sweep_gemv_micro.py` settles everything the pod should not have to debug: all
9 `(micro_size_k, GROUP)` combinations lower to CUDA C, the WQ load width tracks
`micro_size_k` alone (8 -> `uint` 4 B, 16 -> `uint2`, 32 -> `uint4` 16 B), the
index arithmetic is exact, and no combination emits a runtime-indexed local
array. It also holds the shipped path still: at the defaults the emitted CUDA is
byte-identical to what HEAD emits.

## Run order

### 1 — the checkpoint's own `config.json` (2 min, no GPU)

```bash
~/bin/pod 'cat /data00/Qwen3.8-27B-NVFP4/config.json'
```

The go/no-go for the whole model lane, for one command. Cross-check all twelve
fields `_validate_hf_config` gates on: the 7 shape scalars, `layer_types`,
`attn_output_gate`, `rope_theta` (must be 1e7), `partial_rotary_factor` (must
give rotary_dim 64), `rope_scaling` (`rope_type` outside `(None, "default")` or
`factor != 1.0` raises) and `tie_word_embeddings` (must be false). Read the four
`linear_*` fields and the NVFP4 block size (16, vs `pack_fp4`'s default 32) in
the same pass. **`None` means the key is ABSENT**, and an absent key is
validated by nothing — the guards fire only on present-and-wrong.

The guard runs at `model.py:706`, **before** the first `load_file` at `:743`, so
a bad config aborts in about two seconds — not after the ~6-minute CPU dequant.
The reason to read the file first is not the six minutes: a `rope_scaling` block
voids the entire model lane for the day (tileRL implements no RoPE scaling and
refuses rather than serving it wrong), and you want to know that before you
claim a GPU. If it is present and non-default, step 4 is dead for the day,
everything moves to the kernel lane, and that finding is the day's headline —
not a failure.

### 2 — prove the tokenizer imports on the pod's interpreter (3 min, no GPU)

```bash
~/bin/pod 'cd /work/tilerl && PYTHONPATH=src python3 -c "from tilerl.server import get_tokenizer; print(get_tokenizer(\"/data00/Qwen3.8-27B-NVFP4\").encode(\"The capital of France is\"))"'
```

`--selftest` skips the tokenizer entirely, so it does not cover this.
`check_logits` imports `tilerl.server`, which imports fastapi and pydantic at
module level; any ImportError is caught, check 2 silently falls back to three
**synthetic** id sequences, prints `texts ['','','']`, and still records PASS.
That is the exact shape of the defect that voided the baseline's logits.

### 3 — claim GPUs 6 and 7, sync, smoke the harness (5 min)

```bash
~/bin/pod 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
scripts/pod_sync.sh   # ships this checkout to /work/tilerl (GitHub is unreachable from the pod)
~/bin/pod 'cd /work/tilerl && PYTHONPATH=src python3 -u scripts/verify_h20_fp4.py --selftest'
```

Claim **both** 6 and 7 — the two lanes run side by side. Util ~0 **and**
memory ~0: memory at 0% util still means occupied, and a sibling holding memory
has caused OOMs on this box before.

`--selftest` runs all five checks against a CPU tiny fp4 model at B=1 and B=2,
proving the harness imports and the summary prints on the pod's interpreter.
Expect `INFO 1, PASS 4`; the numbers are meaningless. It forces
`TILERL_TARGET=cpu` itself — no GPU pin, no busy probe, so it must never reach a
device. `python3 -u` everywhere: the log is block-buffered otherwise.

**If this fails:** it is the harness or the pod's environment, not the refactor.

### 4 — model lane, detached on GPU 7 (~35 min)

The 27B load is a silent ~6 min of CPU dequant, longer than `tn exec`'s 5-min
no-output timeout, so launch it detached and tail the log:

```bash
# pod_sync.sh tars with --exclude=.git, so /work/tilerl is NOT a repo —
# the commit has to come from the Mac or the report lands with no provenance.
~/bin/pod "cd /work/tilerl && setsid bash -c '
  BENCH_COMMIT=$(git -C /Users/bytedance/code/tileRL rev-parse --short HEAD) \
  PYTHONPATH=src TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \
  python3 -u scripts/verify_h20_fp4.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 --skip 5 \
    --json /work/verify_fp4.json > /work/verify_fp4.log 2>&1
  echo DONE > /work/verify_fp4.done' </dev/null >/dev/null 2>&1 &
  echo launched"
~/bin/pod 'tail -40 /work/verify_fp4.log; ls /work/verify_fp4.done 2>/dev/null'
```

`--skip 5` is the **default**, not the tight-on-time fallback: check 5 costs
10-30 min to attribute a scale-traffic delta, and the kernel lane answers the
same block question in seconds per arm with no model load. `--skip 5` exits 1 by
design — a skipped claim must not print a success.

`TILELANG_CACHE_DIR=/work/tilelang_cache` is the difference between a warm run
and re-paying 30-120 s of NVCC per fresh kernel shape. Exit code is 0 only when
checks 1-4 pass and nothing was skipped.

### 5 — kernel lane: compile + correctness, GPU 6 (8 min, overlapping step 4)

```bash
~/bin/pod 'cd /work/tilerl && CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src TILERL_TARGET=cuda \
  TILELANG_CACHE_DIR=/work/tilelang_cache python3 -u scripts/bench_gemv_micro.py --compile-only'
```

Twelve distinct kernel signatures at 30-120 s of NVCC each is the sweep's
dominant cost, and it is CPU-bound — paying it against the 27B's six-minute
dequant window makes it free.

**HARD RULE: never two timing workloads at once.** Compiles may overlap a load;
nothing may overlap a measurement, or step 4's check-4 headline and the sweep's
microsecond-scale numbers contaminate each other.

### 6 — kernel lane: timing, GPU 6, exclusive (5 min)

Run after step 4's check 4 has printed, on the warm cache:

```bash
~/bin/pod 'cd /work/tilerl && CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src TILERL_TARGET=cuda \
  TILELANG_CACHE_DIR=/work/tilelang_cache python3 -u scripts/bench_gemv_micro.py'
```

Both shapes, six arms, same process, back to back. The **ratio** against the
shipped `(8,4)` arm is the number that decides; the absolute TB/s is the number
that gets compared to Marlin — carefully, see the two gates below.

### 7 — the register question, ptxas (10 min)

The whole ladder rests on whether `micro=32` spills. `ncu` would settle it; so
does `ptxas -v`, at a tenth the cost. Dump the source from an ordinary pod build
and compile it yourself — the incantation is in
[`docs/design-kernels.md`](../docs/design-kernels.md), "Reading the emitted
CUDA". Record registers/thread and local-memory bytes for `(8,4)` and the
winner.

If the winner reports more registers/thread than `(8,4)` despite byte-identical
array shapes, the result stands but the dominance argument does not, and all 9
points have to be measured.

### 8 — verdict and branch (15 min)

- **Gate A passes** → next pod action is a same-process decode A/B through
  `backend.linear_fp4`, the only evidence standard this repo has ever accepted
  for a ship decision. It needs `_CUDA_PLAN`'s K pad moved 256 -> 1024 first
  (`backend.py:100`): a thread block covers `reduce_thread * micro` =
  256/512/1024 elements, so at micro=16/32 a K that is only 256-aligned reads
  past the tensor. Harmless for step 6 — 17408 = 17x1024, 5120 = 5x1024, and
  `bench_gemv_micro.py` asserts alignment rather than padding — but a hard
  prerequisite before micro>8 enters the dispatch plan.
- **Gate A fails on both shapes** → the memory-level-parallelism thesis is dead.
  The next lever is the scale dtype: f32 block scales are 0.25 of tileRL's
  0.75 B/elem, i.e. **a third of the whole fp4 weight stream**, and Marlin's own
  byte definition is 0.5625. That comes before any vendoring decision.
- **Check 2 failed in step 4** → stop the perf program. Every tok/s number is
  denominated on a model whose logits are unvalidated, and that has already
  destroyed one shipped measurement on this checkpoint.

Total: ~75 min of pod wall time to verdicts on both lanes.

## The kernel lane's two gates

One number cannot carry both questions, so do not try.

**Gate A — the thesis, and the only one the sweep can settle.** Does any arm
beat the shipped `(8,4)` by >=5% on **both** shapes, same process, block-16
weights, rel-err <= 1e-2? That is what the kernel's own docstring asks and what
`bench_gemv_micro.py` prints.

**Gate B — the ambition.** Does the best arm get down_proj under **38.9 us**?
That is Marlin's measured wall time on `N=5120 K=17408` at M=1
(`agent-infer/docs/experience/wins/2026-08-23-marlin-nvfp4-decode-bps-tiebreaker.md`).

Use microseconds, not TB/s. Marlin's 1.29 TB/s is computed at 0.5625 B/elem
because NVFP4 stores e4m3 group scales; tileRL stores f32 block scales at
0.75 B/elem, so the same wall time reads **1.33x higher** in tileRL's units. A
kernel that hits "1.30 TB/s in tileRL units" is 1.32x *slower* than the kernel
that number was borrowed from. Matching Marlin's wall time needs 1.72 TB/s in
tileRL units.

Every older sweep in `scripts/` (`_sweep_gemv{,2,3}.py`, `_matrix_gemv.py`,
`bench_gemv_gap.py` before this pass) declared block-**32** scales and still
scored against 0.75 B/elem — a 1.2x overstatement that puts the *baseline* arm
above a 1.30 TB/s bar before anything is tested. `bench_gemv_micro.py` builds
block-16 weights and computes bytes from the tensors.

Gate B will fail. It is knowable before the run: 0.1875 of the 0.75 B/elem is
f32-scale overhead Marlin does not pay, and `micro_size_k` cannot touch a byte
of it. **A pass on A with a fail on B is the expected and useful outcome** — it
says widen the load *and* move the scale dtype, in that order.

## What each check proves, and what a failure means

### 1 — native fp4, no bf16 masters

**Runs AFTER the warmup, on purpose.** `Backend._embed_table_f32` casts the
whole embedding table to f32 on the *first embedding call* and holds it for
process life, so a sample taken at load time misses 4.736 GiB. This check used
to sample early and sum `model.params` only — it would have printed a
comfortable PASS with that tonnage uncounted.

Prints weight bytes by storage class (`.wq` / `.scale` / `.oscale` / `.w8` /
`.wscale` / bare bf16), then resident bytes **by source** — packed, bare bf16,
the f32 embedding cast, the KV pool, the GDN state pool, and the residual
against `torch.cuda.memory_allocated` — plus `memory_reserved`, peak, and
`nvidia-smi` resident, and the allocator delta across the warmup. Both weight
figures are stated: params-only, and params + cast.

- **FAIL "N bf16 masters still resident"** — `keep_master=False` did not take
  and the model is carrying the ~48 GiB it was supposed to delete. The
  refactor's core claim is false. Look at `model.py` `load_hf`'s tail.
- **FAIL "bare tensors > 4.0 GiB"** — same failure seen from the other side
  (the embedding table alone is ~2.4 GiB, everything else should be quantized).
- **"f32 embedding cast" ≈ 4.7 GiB** — expected, not a failure and not gated.
  It is known defect M2 (`docs/experience/2026-08-27-defect-audit.md`); the fix
  is a bf16 `Table` in `_SM90_KERNELS`, which is follow-on structural work.
  A **0** there means the fix landed — or that `Backend._embed_f32` was
  renamed and the harness is reading an attribute that no longer exists. It
  exists today, so a 0 tomorrow is real; check the name before believing it.
- **"KV pool" should read ~1.00 GiB at `--num-blocks 1024`.** The pool is now
  dense over the 16 full-attn layers (audit M3 fixed, `b328ad2`); the script
  reads `kv.k_pool`/`v_pool` directly, so it prints the true number. A 4.00 GiB
  print means the dense packing regressed — investigate before transcribing.
- **The residual is not a cross-check.** "everything else" is defined as
  `allocated − named subtotal`, so the breakdown sums to
  `torch.cuda.memory_allocated()` by construction and a mis-measured source
  just moves into the residual. The one independent signal is the
  `torch allocated X -> Y GiB across the warmup` delta: read it against the
  4.736 GiB cast row, and treat a large unexplained residual as a finding.
- **Nothing is gated on a total.** The 20.3 GiB is COMPUTED, params-only, and
  it assumed every linear is fp4 (the per-channel FP8 ones stay at 1 byte/elem,
  ~47% of the stream). Expect ~22 GiB params, ~27 GiB resident. The gate is
  masters + bare bytes, which is the thing that would actually be a regression;
  the totals are reported so the entry's table can be corrected to measurement.

### 2 — the logits are not void

Greedy-decodes three real prompts with the checkpoint's own tokenizer and
prints the text. **A loader that succeeds is not evidence:** 628c82d served
perfect-looking throughput through the wrong `lm_head` (config said tied, the
checkpoint is untied). Gates on non-degenerate output (not one repeated token,
not two distinct ids) and on the three prompts producing *different*
continuations.

- **FAIL "prompts produced IDENTICAL output"** — the strongest signal that the
  output projection is wrong; a broken `lm_head` collapses to the same
  continuation regardless of prompt. Check `tie_word_embeddings=False` and that
  `lm_head.weight` actually landed.
- **FAIL "degenerate"** — signal is dying somewhere in the stack. Next tool is
  `wins/2026-08-27-zero-centered-rmsnorm.md` (the probe is deleted; the finding is
  the entry), which recorded every rmsnorm output norm and the final
  logits std. Read the raw ids first: greedy decode now stops on the
  tokenizer's `<|im_end|>`/`<|endoftext|>` set, but a healthy model that emits
  one of them inside 8 tokens still trips the "only N tokens generated" arm.
- **WARNING "no tokenizer"** — `tokenizer.json` is missing from the checkpoint
  dir. The gates still run on token ids, but read the ids yourself; do not
  claim the text is sane without seeing it.
- **Text is grammatical but wrong** — that is not a FAIL and the script will
  not catch it. Read the printed continuations. They are the only human check
  in the run.

### 3 — the e4m3 range invariant, on the real kernel

For a sample of real 27B tensors: `max(6*scale)` per tensor (must be inside
e4m3's 448; the renormalization puts it in [6,12)), the fraction of blocks
below e4m3's smallest normal, and the kernel's output against the f32 dequant
reference at M=1 (GEMV), M=8 (w4a8 decode) and M=512 (w4a8 prefill).

**The three arms do not measure the same thing.** The wins entry's 2.3% is a
WEIGHT-path number from a torch sim. Only M=1 is comparable to it, and that
arm dequants in f32, so it should read ~1e-7, not 0.023. M=8 and M=512 are
END-TO-END: the e4m3 weight requant PLUS the per-token e4m3 activation quant
on top of it, so expect ~3-4%. Nothing here isolates the weight path at w4a8.
It is still the first time the real kernel and the real weights are measured —
the CPU tests cannot see e4m3 saturation.

- **FAIL "6*max(scale) SATURATES e4m3"** — `renorm_fp4_scale` is not being
  applied on some load path. Every weight in that tensor above the cap is
  silently clamped and check 2's text will be subtly wrong.
- **FAIL "M=8 fro-relerr > 5%"** — the w4a8 requant is worse on real weights
  than on the simulation. Compare against the M=1 arm: M=1 dequants in f32, so
  if M=1 is clean and M=8 is not, the error is the e4m3 cast, not the nibbles.
- **M=8 error near 3-4%** — as expected: e4m3's 3-mantissa-bit floor on the
  weights and again on the activations. Do not read it as the entry's 2.3%.
- **`6*max` outside [6,12)** — printed as a NOTE, not a failure. It means some
  row's power-of-two split is off by a step; harmless unless it approaches 448.

### 4 — throughput vs the 628c82d baseline

Same method as the baseline entry: steady-state graph replay, 32 ticks,
prefill chunked at 512, pools 1024 blocks / 16384 tokens. Baseline: decode B=1
**19.03 ms/tick = 52.6 tok/s**; prefill 512/2048/8192 = **1947/1847/1773
tok/s**. Also reports decode B=8 (no baseline exists for it) and the implied
weight-stream bandwidth.

**Read the ratio as INFO, not a verdict.** It is a cross-process, cross-day
comparison against a table value from a differently-shaped model, and this box
has already produced a 13% swing from a sibling holding memory at 0% util
(`docs/experience/wins/2026-08-26-batch-decode-h2.md`, note 3). Every
accept/reject in this repo used a same-process A/B. A 0.90-0.95x here is not
by itself something to revert. The old baseline was also partly fast *because*
it was wrong: it re-quantized the per-channel FP8 weights down to 4 bits and
served the wrong `lm_head`.

- **FAIL "decode < 0.95x"** — the gate is a hard `>= 0.95` against that table
  value; 0.90-0.95x is not a revert signal. Do not guess why — see "What the
  verification run does NOT tell us" below.
- **FAIL "decode graph OFF at B=1"** — the tick fell back to eager and the B=1
  numbers are not comparable to the baseline. Rerun. A capture failure at B=8
  alone prints a NOTE and leaves B=1 standing; the flag is snapshotted per
  batch, not read once at the end.
- **prefill < 0.95x** — check 5 does not cover prefill (it measures decode
  only). Prefill regression with clean decode points at the `.oscale` epilogue
  or the N-pad change in `_CUDA_PLAN`, not the scale block.

### 5 — block-16 vs block-32 scales (skipped by default)

The refactor moved the scale block from 32 to the checkpoint's native 16, both
f32. That **doubles scale traffic**: expect ~1.74 → ~3.48 GiB of the weight
stream (the entry's "2.98 → 5.97" assumed all 25.6 G params are fp4; only ~15 G
are — the script computes from the real tensors, so its printed numbers are the
right ones). This check measures decode both ways in the same process, on the
same weights.

**It isolates the scale block and nothing else.** The b32 arm keeps the new
nibble decode, the `.oscale` epilogue, the fp8 arm at 1 B/elem and the new
`_CUDA_PLAN` N-pad.

Arm b16 is the shipped native path (already measured in check 4). Arm b32
re-blocks every block-16 scale grid at load time and re-measures. The re-block
rule: the block-32 scale is `pack_fp4`'s `block_max/6` over the merged pair —
**the max of the two 16-block scales** whenever both blocks reach their grid
top, and the tightest scale that clips nothing otherwise. The mean would push
values past `6*scale` and clamp them, so it is not used. `.oscale` is untouched
because the per-row max is the same either way, so the e4m3 renormalization
survives the re-block (the script prints `6*max(scale)` before and after to
show it).

Read three numbers together:

| number | means |
|---|---|
| scale bytes b16 → b32 | the actual traffic saved (expect ~6 → ~3 GiB) |
| predicted ms/tick saved | that saving divided by the bandwidth arm b16 implies |
| measured ms/tick saved | what block-32 actually bought |

- **measured ≈ predicted** — decode is bandwidth-bound and the block-16 scales
  are exactly where the time went. The memory win has a known price. Do **not**
  retry e4m3 `.scale` storage as the answer: the GEMV issues one scale load per
  8-element micro-tile at block 16 and at block 32, so the decode-instruction
  cost is identical to the 2026-08-25 attempt that lost 5-11%, and only the
  traffic saved changes. Best case it lands near neutral.
- **measured ≪ predicted** — the block change is *not* where the decode time
  goes. Stop optimizing scale traffic; profile the tick
  (`scripts/profile_decode_tick.py`).
- **measured negative (b32 slower)** — the block-32 kernel path is worse for a
  reason unrelated to bytes, e.g. a different `Kp` pad. Note it and move on;
  block-16 is then free of charge.

The script also prints the **added weight error** of the re-block (expect
~3% Frobenius). Arm b32 is a *diagnostic*, not a shippable config at that
error — it exists to attribute the ms, not to propose a change.

Check 5 is destructive (it rewrites the params in place) and therefore always
runs last — checks 1-4 are already recorded by then, so killing the run during
check 5 loses only check 5.

## If you are short on GPU time

```bash
--batches 1           # decode B=1 only (B=1 is always measured — it owns the baseline)
--prefill 512,2048    # drop the 8192 prefill (~5 s per pass plus its own JIT)
--decode-ticks 16     # noisier, half the decode time
--sample 3            # fewer tensors in checks 3 and 5
```

`bench_gemv_micro.py --iters 20` halves the timing pass. Do not cut the arm
list: `(16,2)` is the control on the ladder argument, not filler.

## Failure modes of the harness itself

| symptom | meaning |
|---|---|
| `FATAL: GPU 'N' is not ours` | you asked for one of the user's GPUs. Correct. |
| `FATAL: gpuN is BUSY` | util >10% or >256 MiB resident, sampled twice 3 s apart. 256 because a sibling holding a ~700 MiB context is exactly the case that has OOM'd this box. Wait, or pass `--max-used-mib` / `--max-util` deliberately and own the risk. |
| `FATAL: nvidia-smi failed` | pre-run only: it cannot prove the GPU is free, so it will not run. A transient failure DURING the run prints a warning and reports no smi number; the JSON is written before the closing census, so a 30-second `nvidia-smi` hang at the end cannot cost the run. |
| `FATAL: torch sees N devices` | the `CUDA_VISIBLE_DEVICES` pin did not take (something initialized CUDA first). Refuses rather than risk an allocation on GPUs 0-5. |
| `WARNING: TILELANG_CACHE_DIR unset` | the run works but re-pays NVCC per shape. Set it. |
| a check prints `ERROR` | that check raised; the traceback is above it and the *other* checks still ran. The run is not wasted. |
| `AssertionError: K=... not a multiple of ...` | `bench_gemv_micro.py` on a shape a wider micro cannot cover. Real, not a harness bug — see step 8. |

## After the run

```bash
~/bin/pod 'cat /work/verify_fp4.json' > /tmp/verify_fp4.json   # machine-readable
~/bin/pod 'grep -A30 "== SUMMARY" /work/verify_fp4.log'
```

The final `JSON {...}` line on stdout carries every number in the run. The
wins entry `docs/experience/wins/2026-08-26-native-fp4-w4a8.md` has three
`pending-remote` claims — resident memory, decode ms/tick, and the e4m3
kernel error — and this run closes all three. It is a dated snapshot, so the
numbers go in a **new** entry that links back to it, never over the old one.

## What the verification run does NOT tell us

Four things changed at once: the fp4 scale block (32 → 16), the per-channel FP8
linears (fp4-packed → native e4m3, a **different kernel**), a new `.oscale`
epilogue multiply per linear, and two extra integer ops per nibble in
`_dequant_fp4_macro`. The run produces two decode facts (B=1 and B=8) and
isolates neither. If check 4 comes back at 0.90x, nothing in the harness says
which change did it.

Close that in this order, cheapest first:

1. **Per-op profile of the 27B tick.** `scripts/profile_decode_tick.py` already
   does eager-per-op + graph-total and has never been run on the 27B — every
   per-op number in `docs/` is a 4-layer slice that was all-fp4, block-32 and
   epilogue-free, a mix that no longer exists. ~20 min on the already-loaded
   model, same detached-launch pattern.
2. **fp8 GEMV roofline.** `scripts/bench_fp4_gemv.py` now prints an fp8 arm
   beside the fp4 one (`bench_fp8_shapes`). The fp8 path is ~47% of the decode
   weight stream and has no `%roof` number in any entry. Both arms now compute
   bytes from the tensors instead of a per-elem constant — the old fp4 `%roof`
   figures in `docs/experience` assumed 0.75 B/elem against block-32 weights
   and are ~20% too high. Ratios (PRMT 0.851x, 55.4-vs-57) are unaffected;
   absolute `%roof` across the two scales is not comparable.
3. **`.oscale` cost.** ~257 new epilogue nodes per decode tick, ~2-3% of the
   tick at ~2 µs/node. One timing pass with `oscale` forced to `None`
   (numerically wrong, perf-only) prices it in 2 minutes.

**Do not re-run these — they are settled, with numbers.** fp4 GEMV dequant
mechanism (PRMT 0.851x, MMA 0.504x, the shuffle LUT is within 3% of its own
nodecode floor); register double-buffer, 6-op bitcast decode, byte-LUT, extra
accumulators, noxbuf (all lost); small-M GEMV for B=2..8 (1.56-2.18x slower,
rejected twice); k_split=1 on the WGMMA decode path (ks8 shipped at +7.5%);
bf16-A at B=8 (-13.3% and a gate-failing 3.8% error); e4m3 block scales (-6 to
-11%). A-precision at decode M=1 is settled by physics.

`micro=16/32` used to sit in that list. **It does not belong there** and has
been struck: git shows the rejection text was written for the flat pre-`GROUP`
kernel (`b201ddd` documents `1190885`; `GROUP=4` landed later in `6b39e50`), it
records no ms and no %roof, and the sweep table it points at contains no micro
row at all. Before `GROUP` existed, micro=32 could only be tried at what is now
the `GROUP=4` footprint — 208 register slots against the shipped 52. The
footprint was rejected, never the load width. That is step 6.

**The one correctness thing left.** Check 2 catches a dead `lm_head` and
nothing finer, and `global_divide=True` for the NVFP4 MLP dequant has been an
unchecked assumption since the baseline entry. The baseline run's logits were
void and the fix (`tie_word_embeddings=False`) has never been validated against
a reference framework. If pod time is left, one reference-logit comparison on a
fixed prompt is worth more than another kernel cell — every tok/s number is
void if the model is wrong, and that has already happened once on this
checkpoint.
