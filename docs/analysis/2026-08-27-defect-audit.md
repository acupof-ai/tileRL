# tileRL audit briefing — format/redundancy/residency defects
**HEAD `1a5d8d5`** (audits ran at `10f0b95`; the two commits since touch only `linear()`'s bias device and the fp4 save/load round-trip — **no finding below is stale**). Baseline under attack: **19.03 ms/tick, 52.6 tok/s** at B=1 vs Arle **84.5 tok/s = 11.83 ms/tick**; **20.3 GiB** serving weights on a 96 GiB H20.

**30 findings raised, 1 REFUTED and dropped** (the "training never updates what serves" claim — `load_hf` re-packs `.wq/.scale/.oscale` from the master at `model.py:874-883`, and a measured train→save→load round trip tracks the training loss step for step: 0.091/0.45 at step 4, 0.0018/0.0079 at step 5). **29 confirmed**, of which **5 were the same defect found twice or three times by different auditors** (constant-param casts ⊂ the 931-cast finding; the embedding f32 copy ×2; the GDN state round-trip ×3; paged-attention GQA ×2; `_prefix_state` ×2). **24 distinct defects survive.**

One refutation in thirty is a low kill rate. Read it as: the adversarial pass mostly corrected *magnitudes*, not existence — roughly a third of the surviving findings had their impact cut by 1.5–4x, and two were revised *upward*.

---

## CORRECTNESS — ranked by how silently it fails

### C1. Nothing pins `rope_theta`, `partial_rotary_factor`, `rope_scaling`, or `tie_word_embeddings` to the checkpoint
`_validate_hf_config` (`src/tilerl/model.py:532-540`) cross-checks 7 shape scalars + `layer_types` + `attn_output_gate`. None of the four above. `grep -r rope_scaling src/ tests/ scripts/` returns **zero hits repo-wide** — a YaRN/linear-scaled checkpoint is served as unscaled, silently, where arle refuses to start (`agent-infer/crates/infer-cuda/src/qwen35_load.rs:256-260`).

**This class already fired and voided a shipped measurement.** `docs/experience/wins/2026-08-26-qwen38-27b-baseline.md` says so in its own Surprises: *"This run's logits are void: the config said tied, the checkpoint is untied."* The fix (53cdc87) changed the value and **added no guard**. The corrupting direction is still 100% silent: cfg tied + untied checkpoint pops `lm_head` with `params.pop(..., None)` (`model.py:849-852`), no warning — measured max|Δlogit| 42.0, **top-1 agreement 0/8**. The safe direction raises loudly. `tiny()` is still `tie=True` (`config.py:212`), so copying a preset lands you on the silent side.

Measured sensitivity for the rope half: a 10x-wrong `rope_theta` moves attention logits by **0.23/0.31/0.25/0.36/0.39/0.31 of their std** at positions 128/1k/8k/32k/131k/262k — short prompts are corrupted too, so no smoke test can be safe. Wrong `rotary_dim`: 0.43–0.73.
`rms_norm_eps` is **not** in this class — a 1000x error passes the whole suite and changes rmsnorm output by 4.5e-6 relative at unit RMS. Hygiene, not severity.

Fix: 4 lines in the `checks` dict + a derived tie check (raise if a `lm_head.*` tensor exists while `cfg.tie_word_embeddings`). Caveat: `model.py:661` reads `text_config`; HF multimodal configs put `tie_word_embeddings` at the top level, so a `text_cfg`-only check silently no-ops.

### C2. The gradient gate is inert — 9 of 9 injected gradient corruptions pass
**FIXED 2026-08-27.** The test now calls `backend.cross_entropy_loss_grad`
(the production CE) instead of its local `_ce`, which left a spurious softmax
on the final position and manufactured the 16–106% disagreement that forced
the absolute `< 0.2` tolerance. Tolerance is now `rtol=0.1` (clean worst rel
error 3.6%, measured). Verified: zeroing `gdn_backward`'s `g_q` now fails
`test_production_model_gradcheck` — it passed before.
`tests/test_e2e.py::test_production_model_gradcheck` passed under every poison injected (from outside the tree, via a pytest plugin): residual-branch gradient ×0.5, ×−1 (sign flip), ×0 (no gradient reaches any block), silu gate/up swapped, GDN grads halved, linear weight-grad negated, rmsnorm weight-grad zeroed. **5 of the 9 pass the entire 97-test suite unchanged.** Zeroing any of `gdn_backward`'s 11 grads, or either `attention_gate_bwd` grad, leaves the suite green — 97 passed, 4 skipped, byte-identical to baseline.

Root cause is a live bug **in the gate itself**: its CE helper (`test_e2e.py:412-418`) scatters the −1 into `d[:, :-1]` but leaves a spurious gradient on the final position, so the "analytic" side is not the gradient of the loss it finite-differences — a manufactured 16–106% disagreement, which is what forced the absolute `< 0.2` tolerance at `test_e2e.py:452` against gradients of magnitude 2e-4 to 1e-1. Production is correct (`ops/reference.py:895-899`), so no shipped run has wrong gradients; both backwards measure 6.6e-7 against `torch.autograd`. What is broken is the only thing standing between the next fused/GPU backward and a silent wrong gradient. Fix: call `backend.cross_entropy_loss_grad` from the test, then `rtol=0.1` — with the CE fixed, clean worst rel error is 3.6% and every poison fires.

### C3. Chat template vs. ChatML stop set — the two halves of `server.py` contradict each other
**FIXED 2026-08-27.** `_render_chat` now renders ChatML (`<|im_start|>role\n…
<|im_end|>\n…`), matching the stop set the whole time; gate:
`test_render_chat_is_chatml`. The byte-fallback tokenizer has no eos by
construction, so the tiny/dev path still runs to `max_tokens` — accepted
dev-path behavior, not a serving defect (the real tokenizer carries
`<|im_end|>`).
`server.py:126-131` renders `system: …\nuser: …\nassistant:` plain text; `server.py:64-68` hardcodes `<|im_start|>`/`<|im_end|>` as the entire stop mechanism (`engine.py:605` is the only stop check in the codebase; no config carries an EOS id). No chat template exists in the tree and none is reachable — `tokenizers` has no `apply_chat_template`, and `pyproject.toml` ships no `transformers` and no Jinja engine. Manufactured by 359ce3a, not inherited.

On the **default** serve path (`--model tiny`, what `Dockerfile:47` ships) `ByteTokenizer` has no `stop_token_ids`, so `getattr(..., ())` yields `()` — provably, mechanically, **every request runs all 512 ticks = 9.74 s**, no model behavior required. Zero effect on the 52.6 tok/s number (the bench calls `engine.submit` directly).

### C4. The GDN recurrent state is bf16 where arle carries f32
`kv_cache.py:247` defaults `dtype=torch.bfloat16`; no caller ever overrides it (`engine.py:708-716`, `cli.py:60-68`). arle: `gdr_states: Vec<CudaSlice<f32>>` (`agent-infer/crates/infer-cuda/src/qwen35_state.rs:17`). Measured with the repo's own `reference.gdn_forward` at real 27B GDN dims: **2.4e-3 to 3.6e-3 relative perturbation per token on 48 of 64 layers**, flat in step count (does not compound). Nothing gates it — `tests/test_ops_parity.py:606-632` builds its own f32 state and never touches the pool. Nothing in `docs/` justifies or A/Bs the dtype; it is the constructor default. This is a wrong-format defect that runs at full speed, which is the class the audit was told to hunt.

### C5. `_step_seed` keeps only the seed's low 11 bits
**FIXED 2026-08-27.** `_step_seed` now multiplies the seed by a full-width odd
constant before the XOR/mask (`engine.py`); gate: `test_step_seed_uses_all_seed_bits`
(seeds 1 vs 2049 differ; 10000 seeds give >9990 distinct streams).
`engine.py:59-64`: `((seed << 20) ^ (generated * 2654435761)) & 0x7FFFFFFF` — masking distributes over XOR, so the seed contributes only `(seed & 0x7FF) << 20`. Measured: `len({_step_seed(s,7) for s in range(10000)}) == 2048`; seeds 1 / 2049 / 16385 produce **identical 8-token completions** end-to-end through `build_engine`. The sharp site is not the server (which defaults `temperature=0.0` → argmax, seed unused) but **OPD**: `train.py:161` uses `temperature=1.0, seed=seed+step`, so any run past 2048 steps replays byte-identical teacher rollouts against a prompt list that also cycles — the distillation data silently stops being fresh. The determinism gate compares seed 7 vs 8 and passes. One line.

### C6. Seedless concurrent requests are rank-locked
**FIXED 2026-08-27.** A seedless request now draws `secrets.randbits(31)`
(`server.py`); gate: `test_seedless_requests_decorrelate` (fails with the old
seed=0 default).`server.py:178` gives every seedless request `seed=0`, and `_step_seed` mixes only `(seed, position)`. Measured: three concurrent requests, same prompt, `temperature=0.7`, no seed → **byte-identical 12-token completions**. Different prompts rank-agree 51–97% of positions. Best-of-n and any regenerate button are dead at every temperature. (Correction to the finding: arle is not better here — `sample.rs:203-208` also defaults to 0 — it only decorrelates `n>1` within one request, which tileRL's server does not expose.)

### C7. `clip_grad_norm` clips over a non-parameter
**FIXED 2026-08-27.** `train_step` filters `tape.backward`'s output to param
ids before the clip (`train.py`); gate: `test_pretrain_clips_params_only`
(spies on `clip_grad_norm`, asserts every grad key is a param — fails without
the filter). The backward pass still computes the state grad; a ponytail
marks the upgrade path (non-differentiable state input) if its allocation
ever shows in a peak-memory profile.
`state_gather` is absent from `_BWD`, so the GDN initial recurrent state is a tape leaf and `_linear_attn_chunk` yields a grad for it; `train.py:123` passes the whole dict with no param filter. Measured on tiny: 27 params → 28 grads, over-clip 0.001–6.09% per step, 30 steps diverge by 3.9e-2 relative. At 27B the norm inflation is ~0.1% (negligible) but the **memory** is not: 48 extra f32 tensors, **151 MB per sequence**, all held live through clip + step and each D2H'd to host f64. Also makes the finite-step gate at `train.py:124` depend on a non-parameter. One-line filter at `autograd.py:358`.

### C8. `linear_key_head_dim` is not shape-pinned; the state's K axis silently uses `linear_value_head_dim`
**FIXED 2026-08-27.** `ModelConfig.__post_init__` raises on kd != vd
(`config.py`); gate: `test_linear_head_dim_mismatch_raises`.
`engine.py:708-716` passes one `head_dim` for both trailing axes of the state pool (`kv_cache.py:252-260`), and `reference.py:636-637` re-derives `nkh` from the state's K axis. Executed: `replace(tiny(), linear_key_head_dim=32)` with vd=16 allocates a K axis of 16, runs prefill+decode, and **emits tokens with no exception anywhere**. Latent — every Qwen3.x ships kd=vd=128, and the (nkh, kd) *factorization* is provably inert (a wrong split gives bit-identical output). Fires only on a kd≠vd checkpoint. arle hard-gates `kd == 128 && vd == 128` (`infer-cuda/src/qwen35.rs:289-294`). One-line `__post_init__` raise.

**Umbrella for C1/C4/C8:** the baseline doc admits the NVFP4 dequant convention (`global_divide=True`) is *"assumed, not independently checked against a reference framework's logits."* There is no numerical ground truth for the 27B anywhere in this project. That absence is why every defect in this section is invisible.

---

## PERFORMANCE — decode tick, B=1, against 19.03 ms

| # | defect | mechanism | ms/tick | % of 19.03 |
|---|---|---|---:|---:|
| P1 | **931 real dtype-change nodes per tick** — the residual stream is f32 on a bf16 model | `_SM90_KERNELS = {**_CPU_KERNELS, …}` (`registry.py:87`) overrides only the gemms/attn/gdn, so `embedding`/`rmsnorm`/`rope`/`silu_mul` are the **CPU cell's f32 kernels on sm90**; every linear boundary is bf16-IO. Traced with a `TorchDispatchMode` counting only casts that change dtype: 3 + 16×58 = 931 (353 constant-param recasts, 353 f32→bf16 into linears, 96+96 GDN state/window, 33 int64→int32) | **1.9** (1.3–2.7) | **10%** |
| P2 | GDN state gather/scatter **beyond** the casts | `states[slots, layer_idx]` is advanced indexing (a copy), `index_put_` on the way back — **8 nodes/GDN layer, not 4**; 384 nodes/tick at 48 layers, 288 MiB beyond P1's cast bytes. arle passes `float** state_ptrs` and advances in place (`gdr_decode_batch.cu:127-161`) | **0.4** (0.3–0.6) | 2% |
| P3 | `_epilogue`'s torch multiply | 305 extra graph nodes/tick applying `.oscale` outside the kernel (`backend.py:192-195`); 209 of them also re-widen the fp4 GEMV's already-rounded bf16 output to f32, carrying zero information. The kernel's own ponytail names the fix: fold OScale into the accumulator | **0.3** (0.3–0.6) | 2% |
| P4 | paged-attention re-reads each KV head 6× | grid is `(ceildiv(S,bM), H, B)` with `hkv = hh*Hkv//H` (`kernels_attn.py:80-81`); each of the 6 sibling blocks gathers the whole history itself. Executed index replay: 97.50 MB actual vs 16.25 MB unique = **exactly 6.00x** | **0.05** @ctx≈512 | 0.3% |
| — | | **total recoverable at B=1** | **2.65** (1.9–3.9) | **14%** |

**The arithmetic, honestly: 19.03 − 2.65 = 16.38 ms → 61.0 tok/s → 0.72x Arle.** Best case in the band: 15.07 ms → 66.4 tok/s (0.79x). **It does not reach 11.83 ms. The remaining gap is 4.55 ms and it lives in one place: the fp4 GEMVs.** They are ~83% of the B=1 graph replay (`docs/experience/wins/2026-08-26-decode-tick-profile.md`), i.e. ~14.3 ms of the 27B tick, against a **3.4 ms** fp4-weight-read floor and a **6.16 ms** whole-tick HBM floor (20.4 GB/tick at the measured 3.3 TB/s). tileRL sits at **33% of roof; Arle at 52%.** Even with all 24 defects fixed, the GEMVs must go 14.3 → ~9.7 ms — a **1.47x on dequant issue throughput** — to hit 84.5. Removing every redundant byte this audit found (~1.02 GB of the 20.4 GB tick) lowers the floor itself by only 5%.

**Zero on the c=1 goal** (list them so nobody spends a day on one): sampling's per-row loop (B=1 runs it once, and the sync is unavoidable — the token becomes the next tick's host-side input); the eager `BatchKv` int64 descriptors (192–240 H2D per **eager** tick, but the shipped decode tick is a captured graph whose device int32 buffers make all of them no-ops); ragged mixed-tick padding; the one-prefill-row scheduler; lm_head over all prefill rows. These are prefill / B>1 / TTFT items — real, but denominated in a different metric.

**Worth naming for B=8** (49.1 ms/tick): the GDN round-trip scales to ~6.3 GB/tick (~3.9 ms, 7.9%), and the profile already measures **sampling at 0.394 ms = 8.2% of the slice4 B=8 graph tick** — 8 separate argmax-over-248320 calls plus 8 D2H syncs against a batched sampler that already exists at `reference.py:936-957` and is simply unused. The cast layer is batch-independent, so it drops to 3.8% at B=8.

---

## PERFORMANCE — resident memory, against 20.3 GiB / 96 GiB

| # | defect | cost | note |
|---|---|---:|---|
| M1 | **`_prefix_state` grows without bound** — `engine.py:632-639` clones `states[slot]` + `conv_windows[slot]` every 16 generated tokens; grep shows **2 writers, 0 removers** repo-wide | **4.68 MiB per generated token per row**; 4.57 GiB/1000 tok | Measured: 13 snapshots after 200 tokens, still 13 after the request finished + 200 idle ticks. 80% are provably unreachable (their `PrefixStore` entry was evicted; measured 112 of 140). At B=8 that is 37.4 MiB/tick → OOM inside a single long agent turn. The code's own comment says *"they are small"*; they are 74.81 MiB. **AMENDED 2026-08-27:** this row originally claimed the sibling `self._prefix.insert(...)` pins KV blocks permanently and crashes first with a *"hard `PagedKvPool exhausted` at ~19 requests"*. **That is wrong** — the pinning is self-limiting, because `evict_until_free` (`kv_cache.py:404-407`) runs before every `alloc_block` site (`engine.py:316-318`, `:523-525`). Re-measured at exactly that shape (208-token prompt + 32 generated ≈ 14 blocks/request, `num_blocks=256`): all 60 serial requests complete, free_blocks plateaus at 4/256 from request ~20 on, 84 evictions absorb the pressure, no exception. Only the snapshot leak is real. **FIXED 2026-08-27 (`de0ac27`):** `PrefixStore.on_evict` drops the snapshot when its store entry is evicted; gate: `test_prefix_hit_survives_evicting_its_own_entry`. |
| M2 | **`_embed_table_f32` holds a second, f32 copy of the embedding table** — solely because `kernels.py:446` declares `Table: T.Tensor((V,D), "float32")` and `registry.py:87` never overrides `"embedding"` for sm90 | **+4.736 GiB**, permanent | 248320×5120: bf16 2.368 GiB stays resident (`model.py:886-888` exempts `embed_tokens` from the serving drop) *and* the f32 cast is strongly held on the Backend for process life. Real weights are **25.04 GiB, not 20.3** — invisible to the doc table *and* to `verify_h20_fp4.py:check_memory`, which sums `model.params` only and samples before the warmup that creates the cast. Equals **77,600 tokens of KV**. On training it re-allocates **every optimizer step** (`copy_` bumps `_version`, the cache key). Cannot be deleted after graph capture — fix it upstream: a bf16 `Table`, which is exactly what arle does (`elementwise_basic.cu:228-276`). |
| M3 | **KV pool allocates 64 layer planes; only 16 are ever touched** — `kv_cache.py:98-100` shapes on `num_layers`, indexed by the *global* layer index, while only `cfg.is_full_attn(i)` layers write | **0.75 GiB** at serve default, **3.00 GiB** at the bench's 1024 blocks | **FIXED 2026-08-27.** `PagedKvPool` takes `layer_map` (the global index of each plane), allocates densely, and maps every layer-indexed method through it; a non-full-attn layer raises `KeyError`. Gate: `test_pool_layer_map_is_dense`; full suite green on cpu + metal. |

**Resident status:** M1 (the unbounded leak) and M3 (0.75 GiB) are fixed. Remaining recoverable: M2's 4.736 GiB, which needs the sm90 bf16-IO cell (a bf16 `Table` in `kernels.py:446`). Note the tempting wrong fix: making the state pool f32 to kill P2's casts *costs* +576 MiB and only nets 3.0 MiB/layer/tick — the right fix is reading the bf16 slot in place inside `gdn_decode_fused` (3.15 MB/layer, **half** arle's f32 traffic), with a training carve-out because `linear_attn_chunk`'s recorded `state` argument must stay private under a Tape.

---

## What cannot be settled without the pod

Everything below is derived, estimated, or measured on CPU. Spend GPU time on exactly these, in this order:

1. **The checkpoint's own `config.json`** — `rope_theta`, `partial_rotary_factor`, `rope_scaling`, `tie_word_embeddings`, the four `linear_*` fields, the NVFP4 block size (16, vs `pack_fp4`'s default 32), and the norm-weight dtype (if norms ship f32, 161 of the 931 casts are no-ops and P1 shrinks). A `cat`. Not readable from this host.
2. **One fixed-prompt logits diff against a reference framework** (vLLM/HF, same checkpoint). This is the only instrument that can see C1, C4, C8, or the unvalidated `global_divide=True` convention. It has never been run.
3. **The cost of one baked CUDA-graph node** (1 µs vs 2 µs). The entire P1 band 1.3–2.7 ms is that single unknown. Cheapest possible A/B: capture the same graph with the 385 constant casts hoisted into `materialize`, diff replay time. ~10 minutes, and it prices the whole cast program.
4. **What fraction of 3.3 TB/s the fp4 GEMV achieves, and whether linear time really is ~14.3 ms.** This decides whether anything else in this briefing matters.
5. **Whether L2 absorbs the 6× GQA re-read** at ctx 4k/32k/128k, and whether split-KV (arle's formula gives 20 splits at B=1) beats today's 24-CTA × 64-thread grid — that grid occupies **24 of ~4992 warp slots**. Packing GQA *without* splitting drops it to 4 CTAs and makes B=1 strictly worse; the two must land together.
6. **Resident HBM of `_embed_table_f32`** — `torch.cuda.memory_allocated()` before and after the first decode warmup, which is where `check_memory` currently looks too early.
7. Whether the 27B actually emits ChatML markers (C3's rate), and the `~6 ms` embed-cast docstring vs its own 2.3 ms bandwidth roof.

**Unknown, and stated as unknown:** the true B=1 tick composition at 27B (everything is extrapolated ×16 from slice4); whether bf16 state degrades generation quality (needs real weights); the wall-clock of a 72 MiB unrecyclable `cudaMalloc` every 16 ticks.

---

## Do this first

**Pin the checkpoint as ground truth before any kernel work: read `/data00/Qwen3.8-27B-NVFP4/config.json`, add the four cross-checks to `_validate_hf_config`, and run one fixed-prompt logits diff against a reference framework.** Fifteen minutes of pod time and about twenty lines.

Why it beats the runner-up: **this exact defect class has already destroyed one shipped measurement on this project.** The 19.03 ms / 52.6 tok/s baseline that every number in this briefing is denominated against was measured on a model whose logits the repo's own bench entry calls *void* — tied config, untied checkpoint, full speed, zero errors, top-1 agreement 0/8. That was fixed by changing a value, not by adding a guard, and four surviving findings (rope_theta, rope_scaling, kd≠vd, fp4 block 16 vs 32) have identical shape. Until a reference-logits diff exists, there is no evidence the 27B is computing the right thing at any speed, and the whole perf program is optimizing an unvalidated number.

The runner-up is the right *engineering* move and should be next: **an sm90 bf16-IO kernel cell** — `embedding`, `rmsnorm_partial`/`rmsnorm_apply`, `rope`, `silu_mul` as per-arch entries in `_SM90_KERNELS` (the pattern the gemms already use), plus in-place bf16 state access in `gdn_decode_fused`. One structural change collapses P1 + P2 + M2 at once: ~2.3 ms/tick and 4.7 GiB. But it is 1–2 days, and its payoff is unmeasurable until question 3 above is answered — which is pod work anyway. Do the 15-minute validity check while the pod is warm, then the node-cost A/B, then the cell. The `_prefix_state` eviction hook (M1) is a same-day one-liner that should ride along with any pod session longer than a few hundred tokens, or the session ends in an allocator OOM before it produces a number.