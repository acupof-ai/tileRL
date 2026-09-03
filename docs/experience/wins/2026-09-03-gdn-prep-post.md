# The GDN layer around the WY core is two kernels, not sixty torch ops — H20, 2026-09-03

> Status: Shipped. sm90 default flipped to the WY path, `TILERL_GDN_WY` deleted.
> Gate 1.15x, measured 1.29 / 1.29 / 1.28x on a quiet host.

## Context

The chunkwise-WY prefill core landed faster than fla — 137.4 us/layer against
145.9, where the shipped serial `gdn_chunk_fused` costs ~1400 — and the
27B model-level prefill fell 233.9 → 189.9 ms of GPU-busy, 1.23x. The harness
moved 2238.6 → 2324.6 tok/s, 1.036x.

The gap is not the core. The WY path reached the core through
`reference.gdn_forward`, whose conv1d, SiLU, L2-norm, gate, beta and gated
RMSNorm are torch ops: kernel count went 3827 → 6755, and prefill spends
0.45 s of wall against 0.19 s of GPU-busy. Prefill is host-bound, so 2900
extra launches cost more than 44 ms of saved GPU time bought.

## What Worked

`gdn_chunk_fused` already contains that math; it was split rather than
rewritten. `gdn_prep` emits exactly what the WY kernels consume — q and k
SiLU'd, L2-normed and scaled, v SiLU'd, the cumsum-ready log gate, sigmoid
beta, and the next conv window — and `gdn_post` is the epilogue, gated RMSNorm
times SiLU(z), which had been `rmsnorm` + two casts + `silu_mul`.

Both land as CPU cells first (`kernels.make_gdn_prep` / `make_gdn_post`,
portable f32, no shared memory, reductions serial into a fragment scalar), with
an sm90 override for prep (`kernels_gdn.make_gdn_prep_bf16`: one thread per
head column, the two L2 sums by block allreduce, bf16 out for the WY gemms).
`gdn_post` needs no second schedule — the same kernel is registered with a bf16
IO argument on sm90.

That makes the WY path real on the GPU-less machine: `linear_attn_chunk` now
routes full-length T>1 rows through `Backend._gdn_chunk_wy` on every target,
and the cell without a WY schedule (cpu, metal) uses `reference.gdn_chunk_core`
as the middle stage. `test_gdn_chunk_fused_parity*` stopped being the tautology
its docstring admits to off sm90.

Glue counted by aten dispatch, one GDN layer at the 27B shape (B=1, T=512,
16 key heads, 48 value heads, D=128), the chunkwise core subtracted from both:

| path | torch ops per layer |
|---|---:|
| `reference.gdn_forward` (the old WY route) | 48 |
| `Backend._gdn_chunk_wy` | 2 |

48 is a floor: on CPU the eleven `_f32` casts are no-ops, on CUDA six of them
are real kernels. Working back from the measured 2928-kernel gap over 48 GDN
layers, the old route cost 75 launches a layer, of which 68 were glue; the new
one costs 15.

## Rule

A kernel that beats SOTA inside a torch-eager layer has not shipped anything.
Count launches per layer before believing an end-to-end ratio — on this model
prefill is host-bound, and the glue outweighs the kernel it wraps.

## Results

Every row says how it was obtained. A derived row read as measured is how a
bound turns into a fake measurement.

The whole table comes from one uncontended window (H20, GPU 4, all eight cards
0 MiB / 0% at the start). The shipped-serial arm is not quoted — it is clean
`origin/main` (a702c9a) synced to its own directory and run in the same window,
twice, bracketing the WY arm:

| depth | serial (1st) | WY | serial (2nd) | WY / serial | vs recorded baseline |
|---|---:|---:|---:|---:|---:|
| 512 | 2238.1 | **2887.6** | 2243.2 | 1.287x | 1.290x |
| 2048 | 2215.4 | **2852.5** | 2220.7 | 1.284x | 1.287x |
| 8192 | 2135.3 | **2729.0** | 2146.8 | 1.271x | 1.274x |

The serial bracket closes to 0.2-0.5% either side of the WY arm and lands
within 0.3% of the recorded 2238.6 / 2215.9 / 2142.4, so nothing drifted across
the window. The WY numbers reproduce an earlier session to 0.15% / 0.06% /
0.14% (2891.7 / 2854.2 / 2732.8). Ship gate was 1.15x.

| metric | how | shipped serial | WY, torch glue | WY + prep/post |
|---|---|---:|---:|---:|
| prefill GPU-busy, ms (64 layers, 512 tok) | measured | 233.8 | 189.9 | **176.4** |
| kernels per prefill | measured | 3827 | 6755 | **3925** |
| GDN launches per layer | derived from the row above | 14 | 75 | 16 |
| WY core, us/layer (fla 145.5) | measured | ~1400 | 137.4 | 121.8 |

**The prediction landed, and where it missed.** The launch-count argument
rested on ~3875. Actual 3925 — **+50, +1.3%**, or 16 launches a layer where 15
was predicted: one call a layer unaccounted. The 1.77x launch inflation that
made a 1.23x GPU-busy win read as 1.036x end to end is gone (3925/3827 =
1.026x). `gdn_prep` costs 3.34 ms over 48 launches and `gdn_post` 5.35 ms over
48 — 8.7 ms of GPU for what had been ~36 ms of torch glue; `rmsnorm_fused`
drops 209 → off the top eight and `silu_mul` 112 → 64, both absorbed. The path
also beats fla's own chunked route end to end: 176.4 ms and 3925 kernels
against 193.4 ms and 6899.

### Correction: the 8192 spread was not the neighbouring tenant

An earlier revision of this entry marked 2729.0 approximate and blamed 27.0%
spread on a tenant's job restarting mid-run. **That attribution was wrong**, and
the quiet window disproved it: with the host empty at the start the same row
came back at 50.4% spread, while the serial arm at the same depth in the same
window read 0.0% and 1.2%. Host contention does not explain a spread that only
one arm sees.

Measuring the distribution instead of the summary settles it. `suite_prefill`
reports a **median of three runs** and `(max-min)/median` as spread, on one
engine reused across all three lengths. Timing eight consecutive runs at each
depth directly:

| arm | 8192, eight consecutive runs | spread |
|---|---|---:|
| WY | 1217 1215 1216 1216 1217 1217 1216 1215 | **0.16%** |
| serial | 940 940 939 940 939 941 939 939 | 0.21% |

(That pass ran under contention — GPUs 0 and 2 at 98% — so the absolutes are
~2.2x low and only the shape is being read. The ratio there is 1.294x.)

The WY path at 8192 is the *steadiest* series in the set, not a noisy one. The
27% and 50% figures are an artifact of a three-run sample inside a suite that
reuses one engine across three lengths; they are not a property of this change.
The median they report is reproducible to 0.14% across two independent
sessions. **2729.0 stands without qualification.** What remains unexplained is
why the artifact lands on the WY arm and not the serial one — plausibly because
WY's per-tick GPU time is lower, so a fixed host-side stall is a larger
fraction of it, but that is a hypothesis I did not test.

The 8192 ratio has now been measured five times across three sessions and two
contention states: **1.273, 1.286, 1.282, 1.271, 1.294** — the claim that
survived contention, and the one that survives the quiet host too.

### Baseline file

`docs/experience/wins/bench-baseline.json` is deliberately not hand-edited; it
still holds the pre-flip 2237.8 / 2218.4 / 2144.8. From this run the harness's
own raise would set **len512 → 2887.6** and **len2048 → 2852.5** (it emitted
both `RAISED` lines). It declined to raise len8192 because that row's sampled
spread exceeded its noise bar; on a quiet card that row should seed at ~2729.
Until a post-merge `tilerl bench --suite prefill` runs, the prefill gate is
loose — a regression to ~2300 tok/s would still pass against the stale
baseline.

## Correctness

sm90 parity, `scripts/probe_gdn_wy.py` on H20 (GPU 3, verified 0 MiB before the
run), gate 1e-2 relative:

| check | out | state | window |
|---|---:|---:|---:|
| (b2) `linear_attn_chunk` (WY) vs `reference.gdn_forward` | 4.0e-3 | 5.1e-3 | 0 |
| (b) `_gdn_wy_core` vs `reference.gdn_chunk_core` | 6.6e-3 | 6.1e-3 | — |
| (b) `_gdn_wy_core` vs fla `chunk_gated_delta_rule` | 3.5e-3 | 4.8e-3 | — |

(b2) is the gate for the two new cells — the only thing that runs
`make_gdn_prep_bf16` and the bf16 `gdn_post` at all. It used to build its inputs
at amplitude 0.5 and read 5.9e-3 / 7.7e-3. Half the amplitude of the CPU gate is
not a gate: `test_gdn_chunk_fused_parity_full_scale` exists because a pipeline
that passed at 0.1 was 26% wrong at 1.0
([2026-08-25-gdn-chunked-gdr-rejected.md](../errors/2026-08-25-gdn-chunked-gdr-rejected.md)).
Raised to 1.0, where the row above was measured.

### The 5.9e-3 was attributed to bf16; here it is decomposed

Nothing separated `make_gdn_prep_bf16` — the newest code in the change, with a
hand-rolled block allreduce and a GQA write predicate — from the six transcribed
WY kernels, and a 1e-3 error inside it reads as bf16 noise at the layer level.
`reference.gdn_prep` is now the front half of `gdn_forward` rather than a copy of
it, so probe stage (b1) compares the cell's six outputs against the spec, at
amplitude 1.0:

| gdn_prep output | dtype | rel |
|---|---|---:|
| `qn` | bf16 | 2.1e-3 |
| `kn` | bf16 | 2.6e-3 |
| `v` | bf16 | 2.8e-3 |
| `g` | **f32** | **8.3e-8** |
| `beta` | bf16 | 2.0e-3 |
| window | f32 | **0** |

bf16 keeps 8 mantissa bits, so its own rounding unit is 2^-9 = 1.95e-3. Every
bf16 output lands within 1.5x of that floor, and the two outputs the cell does
*not* round — the log gate and the carried conv window — are exact to f32 and
bit-exact respectively. The gate and the window run the same conv, softplus and
window-slice arithmetic as the rest of the kernel, so their accuracy is what
rules out an arithmetic fault and leaves the format. `qn` and `kn` are the two
that go through the block allreduce; both sit at the floor.

The layer's 4.0e-3 is therefore not prep-dominated: prep contributes ~2-3e-3 at
the floor, and the WY core carries the remainder (6.6e-3 against the f32
reference, 3.5e-3 against fla — two independent bf16 implementations of the same
algorithm differing by that much *is* the format).

Per-kernel parity of the WY core against fla 0.5.2: `kkt` 4.6e-4 max abs,
`solve_tril` 4.9e-4, `w` 2.4e-4, `u` 0, `h` 9.8e-4, `V_new` 9.8e-4, final state
4.7e-4, `o` 2.3e-4. State-scan known-answer rows exact.

Two guards the numbers above cannot produce. `make_gdn_prep_bf16` binds one
thread to a head column of q, k and v alike, so it needs key_dim == value_dim;
the f32 cell loops each extent separately, so the CPU parity gate is structurally
blind to it and a mismatched checkpoint would write past `Qo` rather than return
a wrong number. And `gdn_state_scan` sizes `h` by `S // chunk` while looping
`ceildiv`, so a tail chunk writes past it — the routing predicate was the only
thing standing in the way. Both are asserts now, in `_gdn_prep` and
`_gdn_wy_core`.

Parity on the GPU-less machine, `Backend.linear_attn_chunk` vs
`reference.gdn_forward`'s per-step scan, tiny model at full input scale:
cpu out 2.1e-6 / state 4.3e-7 / window 0, metal 7.5e-7 / 4.9e-7 / 0.
`TILERL_TARGET=cpu uv run pytest -q`: 170 passed.

The 137.4 → 121.8 us/layer on the core is not a schedule change:
`gdn_chunk_core_tl` cast q, k, v and beta on every call and `gdn_prep` now
emits them already cast, leaving only the 3.0 us state cast inside the core.

## What did not change

The serial `gdn_chunk_fused` stays and is still reachable: speculative verify
(`keep_steps`), ragged rows, and any length that is not a whole multiple of the
64-token chunk. That is a capability boundary, not a second path to the same
answer.
