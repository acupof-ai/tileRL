# H20 pod verification — native fp4 w4a8

Run order for tomorrow. One script does all five checks
(`scripts/verify_h20_fp4.py`); everything else here is setup and reading the
output. Dev-only tooling — no bench entry for the harness itself, but the
numbers it prints are what the wins entry gets updated with.

**Non-negotiable:** GPUs 0-5 are the user's own training run, 100% util and
~94 GiB each. Only **6 and 7** are ours. The script refuses to start on
anything else and refuses a GPU that is busy — do not talk it out of that.

---

## Before you start — four things, ~3 minutes, all before the 6-minute load

1. **Read the checkpoint's own `config.json`.**

   ```bash
   ~/bin/pod 'cat /data00/Qwen3.8-27B-NVFP4/config.json'
   ```

   `_validate_hf_config` now hard-raises on `rope_theta != 1e7`, a
   `partial_rotary_factor` that does not give rotary_dim 64, any `rope_scaling`
   whose `rope_type` is outside `(None, "default")` or whose `factor != 1.0`,
   and `tie_word_embeddings != false` — and it raises **after** ~360 s of CPU
   dequant. A YaRN block at `max_position_embeddings=262144` is entirely
   plausible on this checkpoint. Read those five, plus the four `linear_*`
   fields and the NVFP4 block size (16, vs `pack_fp4`'s default 32), then
   launch. The harness prints the five itself just before the load: **`None`
   means the key is ABSENT**, and an absent key is validated by nothing — the
   guards fire only on present-and-wrong.

2. **Prove the tokenizer imports on the pod's interpreter.**

   ```bash
   ~/bin/pod 'cd /work/tilerl && PYTHONPATH=src python3 -c "from tilerl.server import get_tokenizer; print(get_tokenizer(\"/data00/Qwen3.8-27B-NVFP4\").encode(\"The capital of France is\"))"'
   ```

   `--selftest` skips the tokenizer entirely, so it does not cover this.
   `check_logits` imports `tilerl.server`, which imports fastapi and pydantic
   at module level; any ImportError is caught, check 2 silently falls back to
   three **synthetic** id sequences, prints `texts ['','','']`, and still
   records PASS. That is the exact shape of the defect that voided the
   baseline's logits.

3. **Commit before syncing.** `pod_sync.sh` tars the working tree
   (`--exclude=.git`) while §2's launch line stamps the JSON with
   `git rev-parse --short HEAD` — with anything uncommitted the report names a
   commit that does not contain what ran. Commit, or pass
   `BENCH_COMMIT=<sha>-dirty`.

4. **Budget the window: `--skip 5` on the first pass.** Check 5 costs 10-30 min
   to attribute a scale-traffic delta worth ~1.7 GiB of a 20.4 GB tick. The
   first-order question — is linear time really ~14.3 ms, and what fraction of
   3.3 TB/s does the fp4 GEMV reach — decides whether anything else in the
   audit matters, and this harness does not answer it at all.
   `scripts/profile_decode_tick.py` does, in ~20 min on the already-loaded
   model, and has never been run on the 27B; `bench_fp4_gemv.py` answers the
   block question in 30 s. `--skip 5` exits 1 by design.

## 0. Claim a GPU and sync

```bash
# what's free right now
~/bin/pod 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
```

Pick 6 or 7 with util ~0 **and** memory ~0. Memory at 0% util still means
occupied — a sibling holding memory has caused OOMs on this box before.

```bash
# ships this checkout to /work/tilerl (GitHub is unreachable from the pod)
scripts/pod_sync.sh
```

## 1. Smoke the harness before spending GPU time (~1 min, no GPU)

```bash
~/bin/pod 'cd /work/tilerl && PYTHONPATH=src python3 -u scripts/verify_h20_fp4.py --selftest'
```

Runs all five checks against a CPU tiny fp4 model at B=1 and B=2. Proves the
harness imports, the checks execute, the re-block math runs and the summary
prints **on the pod's interpreter**. Expect `INFO 1, PASS 4`; the numbers are
meaningless (tiny model, no baseline). `--selftest` forces `TILERL_TARGET=cpu`
itself — it has no GPU pin and no busy probe, so it must never reach a device.
`python3 -u` everywhere: the log is block-buffered otherwise and `tail` lags.

**If this fails:** it is the harness or the pod's environment, not the
refactor. Fix it here before touching the 27B — a failure after a 6-minute
load costs 6 minutes.

## 2. The one command (~20-35 min; first run longer, NVCC)

The 27B load is a silent ~6 min of CPU dequant, longer than `tn exec`'s 5-min
no-output timeout, so launch it detached and tail the log:

```bash
# pod_sync.sh tars with --exclude=.git, so /work/tilerl is NOT a repo —
# the commit has to come from the Mac or the report lands with no provenance.
~/bin/pod "cd /work/tilerl && setsid bash -c '
  BENCH_COMMIT=$(git -C /Users/bytedance/code/tileRL rev-parse --short HEAD) \
  PYTHONPATH=src TILERL_TARGET=cuda TILELANG_CACHE_DIR=/work/tilelang_cache \
  python3 -u scripts/verify_h20_fp4.py /data00/Qwen3.8-27B-NVFP4 --gpu 7 \
    --json /work/verify_fp4.json > /work/verify_fp4.log 2>&1
  echo DONE > /work/verify_fp4.done' </dev/null >/dev/null 2>&1 &
  echo launched"
```

Then poll:

```bash
~/bin/pod 'tail -40 /work/verify_fp4.log; ls /work/verify_fp4.done 2>/dev/null'
```

`TILELANG_CACHE_DIR=/work/tilelang_cache` is the difference between a warm run
and re-paying 30-120 s of NVCC per fresh kernel shape. The script sets it
itself if `/work` exists, but set it explicitly anyway. `--gpu 7` is also
what the script defaults to; pass `--gpu 6` if 7 is taken.

Exit code is 0 only when checks 1-4 pass and nothing was skipped. Check 5 is
recorded `INFO`, not `PASS`: it reports a delta, it asserts nothing. A `SKIP`
exits 1 by design — `--skip 5` must not print a success for a claim that was
never tested.

## 3. What each check proves, and what a failure means

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
- **"KV pool" over-reads by 4x.** The pool is shaped on all 64 layers
  (`kv_cache.py:98`) and only the 16 full-attention planes are ever written
  (audit M3), so at `--num-blocks 1024` the line prints 4.00 GiB where 1.00 is
  reachable. Nothing gates on it and nothing OOMs; annotate the number when
  transcribing it into the wins entry.
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
  `scripts/diag_slice.py`, which prints every rmsnorm output norm and the final
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

**Read the ratio as a smoke test, not a verdict.** It is a cross-process,
cross-day comparison against a table value, and this box has already produced a
13% swing from a sibling holding memory at 0% util
(`docs/experience/wins/2026-08-26-batch-decode-h2.md`, note 3). Every
accept/reject in this repo used a same-process A/B. A 0.90-0.95x here is not
by itself something to revert. The old baseline was also partly fast *because*
it was wrong: it re-quantized the per-channel FP8 weights down to 4 bits and
served the wrong `lm_head`.

- **FAIL "decode < 0.95x"** — **read this verdict as INFO.** The baseline is a
  differently-shaped model (628c82d kept bf16 masters resident, requantized the
  per-channel FP8 weights to 4 bits, and served `embed_tokens` as `lm_head`),
  and the gate is a hard `>= 0.95` against a table value from another day.
  0.90-0.95x is not a revert signal. Do not guess why: check 5 runs next and
  attributes one of the four changes (see §7).
- **FAIL "decode graph OFF at B=1"** — the tick fell back to eager and the B=1
  numbers are not comparable to the baseline. Rerun. A capture failure at B=8
  alone prints a NOTE and leaves B=1 standing; the flag is snapshotted per
  batch, not read once at the end.
- **prefill < 0.95x** — check 5 does not cover prefill (it measures decode
  only). Prefill regression with clean decode points at the `.oscale` epilogue
  or the N-pad change in `_CUDA_PLAN`, not the scale block.

### 5 — THE PERF RISK: block-16 vs block-32 scales

The refactor moved the scale block from 32 to the checkpoint's native 16, both
f32. That **doubles scale traffic**: expect ~1.74 → ~3.48 GiB of the weight
stream (the entry's "2.98 → 5.97" assumed all 25.6 G params are fp4; only ~15 G
are — the script computes from the real tensors, so its printed numbers are the
right ones). This check measures decode both ways in the same process, on the
same weights.

**It isolates the scale block and nothing else.** The b32 arm keeps the new
nibble decode, the `.oscale` epilogue, the fp8 arm at 1 B/elem and the new
`_CUDA_PLAN` N-pad. It does not answer "what did the refactor cost" — see §7.

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

## 4. If you are short on GPU time

```bash
--skip 5              # drop the A/B; exits 1, and check 5 shows SKIP in the summary
--batches 1           # decode B=1 only (B=1 is always measured — it owns the baseline)
--prefill 512,2048    # drop the 8192 prefill (~5 s per pass plus its own JIT)
--decode-ticks 16     # noisier, half the decode time
--sample 3            # fewer tensors in checks 3 and 5
```

Budget check 5 generously: it pushes all 497 fp4 tensors through `pack_fp4`
(~32 bytes allocated per weight element, row-chunked), then reloads and
re-times. On its own that is plausibly 10-30 min. If the day is tight, run the
first pass with `--skip 5` and get the block answer from `bench_fp4_gemv.py`
instead — same question, 30 s of GPU, no model load.

Check 5 is destructive (it rewrites the params in place) and therefore always
runs last — checks 1-4 are already recorded by then, so killing the run during
check 5 loses only check 5.

## 5. Failure modes of the harness itself

| symptom | meaning |
|---|---|
| `FATAL: GPU 'N' is not ours` | you asked for one of the user's GPUs. Correct. |
| `FATAL: gpuN is BUSY` | util >10% or >256 MiB resident, sampled twice 3 s apart. 256 because a sibling holding a ~700 MiB context is exactly the case that has OOM'd this box. Wait, or pass `--max-used-mib` / `--max-util` deliberately and own the risk. |
| `FATAL: nvidia-smi failed` | pre-run only: it cannot prove the GPU is free, so it will not run. A transient failure DURING the run prints a warning and reports no smi number; the JSON is written before the closing census, so a 30-second `nvidia-smi` hang at the end cannot cost the run. |
| `FATAL: torch sees N devices` | the `CUDA_VISIBLE_DEVICES` pin did not take (something initialized CUDA first). Refuses rather than risk an allocation on GPUs 0-5. |
| `WARNING: TILELANG_CACHE_DIR unset` | the run works but re-pays NVCC per shape. Set it. |
| a check prints `ERROR` | that check raised; the traceback is above it and the *other* checks still ran. The run is not wasted. |

## 6. After the run

```bash
~/bin/pod 'cat /work/verify_fp4.json' > /tmp/verify_fp4.json   # machine-readable
~/bin/pod 'grep -A30 "== SUMMARY" /work/verify_fp4.log'
```

The final `JSON {...}` line on stdout carries every number in the run. The
wins entry `docs/experience/wins/2026-08-26-native-fp4-w4a8.md` has three
`pending-remote` claims — resident memory, decode ms/tick, and the e4m3
kernel error — and this run closes all three. It is a dated snapshot, so the
numbers go in a **new** entry that links back to it, never over the old one.

## 7. After verification passes — what the run does NOT tell us

Four things changed at once: the fp4 scale block (32 → 16), the per-channel FP8
linears (fp4-packed → native e4m3, a **different kernel**), a new `.oscale`
epilogue multiply per linear, and two extra integer ops per nibble in
`_dequant_fp4_macro`. The run produces three decode facts (B=1, B=8, and check
5's block A/B) and check 5 isolates only the first. If check 4 comes back at
0.90x, nothing in the harness says which change did it.

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
accumulators, noxbuf, micro=16/32 (all lost); small-M GEMV for B=2..8 (1.56-2.18x
slower, rejected twice); k_split=1 on the WGMMA decode path (ks8 shipped at
+7.5%); bf16-A at B=8 (-13.3% and a gate-failing 3.8% error); e4m3 block scales
(-6 to -11%). A-precision at decode M=1 is settled by physics.

**Highest-EV next experiment** (kernel-level, no model load, ~15 min): in
`kernels_linear.py:342` each thread's four W loads are 4 bytes each, 128 bytes
apart. Re-partition K so a thread's GROUP micro-tiles are contiguous
(`base = kg*GROUP*block_K + kr*GROUP*micro_size_k + g*micro_size_k`) and
vectorize W across the group: one 16-byte load, identical DRAM traffic and warp
coalescing, 4x fewer W-load instructions, and the per-micro-tile scale lookups
drop from 4 to 2 at block 16. The mechanism is not a guess — `linear_fp8_gemv`
loads W as a 128-bit vector and beats the fp4 GEMV while reading 1.65x the
bytes. This is **not** the twice-rejected `micro=16/32`: `micro_size_k` and
every register buffer stay the same size. Gate: >=5% at N=34816/K=5120 and
N=5120/K=17408, fro-relerr <=1e-2, then a same-process decode A/B before
shipping. Fallback cell in the same sweep: grid K-split (ks in 2/4/8, f32
atomics into a zeroed Y) — never executed on this kernel, and the identical
lever just won +7.5% on the sibling WGMMA path.

**The one correctness thing left.** Check 2 catches a dead `lm_head` and
nothing finer, and `global_divide=True` for the NVFP4 MLP dequant has been an
unchecked assumption since the baseline entry. The baseline run's logits were
void and the fix (`tie_word_embeddings=False`) has never been validated against
a reference framework. If pod time is left, one reference-logit comparison on a
fixed prompt is worth more than another kernel cell — every tok/s number is
void if the model is wrong, and that has already happened once on this
checkpoint.
