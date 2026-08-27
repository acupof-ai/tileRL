# Is the 1.6x gap to Arle the price of TileLang? — evidence, 2026-08-27

> Status: Answered — no. Three corrections to the premise, one lever identified.
> Addendum 2026-08-27: the register-resident-B question this brief named as
> decisive is settled — **CAN**, proven by emitted CUDA C. See "The
> register-resident verdict" below.

## The answer

**The gap is not the language.** Arle's own hand-written native-CUDA scalar FP4
GEMV — three rewrite generations (constant table -> bit manipulation -> PRMT byte
lookups), 7.5x on the kernel, ncu-tuned to within 12% of its own instruction
floor — landed at **52.3 tok/s** c=1 on 1xH20. tileRL's TileLang fp4 GEMV lands
at **52.6 tok/s** on the same box and checkpoint. Writing the same kernel
*algorithm* in raw `.cu` instead of TileLang bought Arle **0.994x**.

Arle got past 52.3 by **vendoring**, not by writing better CUDA. Its own
retrospective: *"The kernel that solved this had been vendored in the tree the
whole time, one kFE2M1f instantiation away. Three sessions of hand-optimizing
the scalar GEMV produced a real 7.5x and still could not reach what the existing
SOTA kernel gave immediately."*
(`agent-infer/docs/experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md`)

## Three corrections to the premise

**1. "Arle is all raw .cu" is false.** Arle ships **4,967 lines of TileLang
Python** generating 47 AOT-compiled CUDA kernels. Its own README: *"The CUDA
feature uses TileLang as the only AOT compiler surface for paged attention and
Qwen3.5 chunk-wise GDR."*
(`agent-infer/crates/cuda-kernels/tools/tilelang/README.md:1-6`, `kernels.toml`)

Caveat that cuts the other way: on the shipped 84.5 c=1 path, **zero TileLang
executes**. The TileLang paged-attention lane is BF16-pool only and is excluded
from graph capture; the 84.5 config runs `--kv-cache-dtype fp8`, so decode takes
the native quantized kernel first. The TileLang GDR lane is gated `seq_len > 1`.
(`qwen35_attention.rs:941-945, :636-641, :2338-2342`)

**2. Arle's Qwen3.8 NVFP4 decode does not use INT8 tensor cores.** I claimed it
did. `nvfp4_to_w4afp8.cu`'s only caller is the DeepSeek-V4 MoE loader
(`dsv4/load.rs:1392`) — a different checkpoint with E8M0 block scales. Qwen3.8
keeps its E2M1 nibbles and runs Marlin `kFE2M1f` with **BF16** activations on
BF16 tensor cores: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
(`marlin_gemm.cu:9-13`, `marlin_template.h:92`)

**3. Arle's Qwen3.8 attention is not FlashMLA.** I claimed it was. `flashmla`
appears only under `dsv4/*`, never in any `qwen35_*.rs`. Decode attention is
Arle-authored `paged_attention_quantized_fa3.cu` (602 lines).

## What Arle actually runs at c=1

Every dense projection is Marlin-repacked at load, and it is mandatory — a shape
Marlin cannot take **fails the load** rather than degrading to a scalar arm, and
the pre-repack bytes are freed (`loader.rs:2781-2810`, `quant_linear_fp4.rs:19-26`).

| component | provenance | LOC |
|---|---|---:|
| all 7 linears + lm_head | **vendored Marlin** (Frantar / Neural Magic / IST-DASLab) | 3,903 (2,492 of it in `.h`/`.hpp`, invisible to a `.cu` line count) |
| Arle's Marlin shim | authored | 407 + 148 |
| decode prep (q/k RMSNorm + partial RoPE + paged KV write, **one launch**) | authored | 250 |
| paged attention (quantized pool) | authored | 602 |
| conv1d + gated-delta recurrence | authored | 333 + 758 |

Arle's ladder to 84.5: 9.3 -> **52.3** (hand-written PRMT GEMV) -> **57.9**
(Marlin, +11%) -> 60.2 (blocks-per-SM search) -> 66.6 (down/o repack) -> **~83**
(decode CUDA graph restored, **+25%**) -> 84.5 (bps tiebreaker). Marlin was not
the big step; the captured graph was.

## The lever

**`micro_size_k = 8` at `src/tilerl/ops/kernels_linear.py:314`.** It makes the
fp4 decode GEMV load 4 bytes of weight per thread — `LDG.32`. Its two siblings,
same split-K schedule, same warp allreduce, same author, issue `LDG.128`:

| kernel | bytes/thread | % of roof |
|---|---:|---:|
| `make_linear_bf16_gemv` (`:433`) | 16 | 42–116% |
| `make_linear_fp8_gemv` (`:474`, micro=16 x 1B) | 16 | — |
| **`make_linear_fp4_gemv`** (`:341`, micro=8 x 0.5B) | **4** | **24–33%** |

The kernel's own docstring records micro=16/32 as "tested worse" — and that
rejection is **stale**. Git shows the text first appears in `b201ddd`, the docs
commit for the FLAT pre-`GROUP=4` kernel (`1190885`); `GROUP = 4` landed
afterwards in `6b39e50`, and the sentence was carried verbatim through the
`197b966` file split. It preserves no ms and no %roof anywhere. The old negative
was most likely a `GROUP=4 x micro=32` register spill (`ws[4][32]` = 128 f32) —
which is an argument for `micro=32, GROUP=1`, not against widening.

## What is already stale at HEAD

The audit's "83% of the B=1 tick is fp4 GEMVs", and Evidence A's "tileRL
re-quantizes Arle's FP8 half down to fp4", are **both stale**. Since `10f0b95`
the per-channel FP8 half of the checkpoint is served native
(`model.py:828-841`, `registry.py:104`, `backend.py:101`). That is three commits
after `628c82d`, the baseline every number in this repo is denominated against.
**Nobody has re-benched.** HEAD serves a different weight program than the
measurement it is judged by.

## The register-resident verdict: CAN, proven by codegen

Can TileLang express Marlin's register-resident quantized B operand — the thing
that makes Marlin's dequant free — or does the remaining 1.5–1.85x require
vendoring `.cu`? Settled 2026-08-27 on this GPU-less Mac via the
`scripts/cuda_codegen.py` shim: **CAN.**

- The primitive is `T.gemm`'s SR variant: A in shared, **B in a fragment**
  (`is_gemm_sr()`, `tilelang/tileop/gemm/gemm_base.py:57`; CUDA lowering at
  `tilelang/cuda/op/gemm/gemm_mma.py:143-172` — no `ldmatrix_b`, the fragment
  goes straight to the tensor core).
- A tileRL-shaped fp4 decode kernel whose dequantized weight never touches
  shared memory emitted 7,833 bytes of CUDA C. Its
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` is
  character-for-character Marlin's instruction (`marlin_template.h:91-95`), B
  operand read from a register array in both.
- **Correction to this brief's premise**: "every tilelang dequantize_gemm
  example materializes `B_dequantize_shared`" is false. The canonical documented
  pattern is register-resident (`examples/dequantize_gemm/README.md:33`,
  `example_dequant_gemm_w4a8.py:145`). The two examples that materialize it are
  both WGMMA kernels, where shared B is a hardware requirement — and tileRL
  copied one of them by name (`kernels_linear.py:232`). The anti-pattern was
  inherited from the wrong donor, not imposed by the language.
- Hopper is not a blocker: a fragment B makes `CheckWgmma` fail, so instruction
  selection falls through to `cuda.mma` — exactly what Marlin runs on H20.
- The one real cost is target neutrality: **Metal does not implement SR**
  (`gemm_metal.py:97-98`, SS only). A Marlin-shaped kernel is a per-arch sm90
  cell, which `_SM90_KERNELS` already is.
- Size of the prize from tileRL's own constants: the live decode kernel
  (`make_linear_fp4_fp8_mma`, `_CUDA_PLAN`'s `linear_fp4_fp8_decode`) spends
  **12 KiB/CTA** on `W_shared` (4 KiB e4m3 × 3 stages) against `WQ_shared`'s
  2 KiB — shared memory buying nothing but a format conversion. (The
  unreachable `make_linear_fp4_mma` spends 24 KiB in bf16.) Freeing it is an
  occupancy lever; this repo measured +7.5% from one of those in `5d53d9b`.

Instrument honesty: the high-level `T.gemm` SR path could not be lowered on
this wheel — `RegisterGemmImpl` takes raw C++ function pointers with no FFI, so
`tl.gemm` is unreachable for *any* operand scope here (a shared-B control
failed with byte-identical text; the failure carries no information about
operand scope). The warp-level route, which bypasses `tl.gemm`, is what
lowered. The pod should confirm the high-level path emits the same operand
structure.

## Recommendation

Ordered, with the cheap disambiguating measurements first.

0. **Re-baseline HEAD** (0.5 day, pod). `628c82d` was 25.62 G params all fp4 at
   block-32 = 16.01 GB/tick. HEAD is 14.97 G fp4 at block-16 f32 scales
   (11.23 GB) + 10.65 G native e4m3. Different program, unmeasured.
1. **bf16 block scales on the fp4 path** (0.5 day, ~5 lines). At block 16, f32
   scales are 0.25 B/elem — **a third of the entire fp4 weight stream**.
2. **`micro_size_k` 8 -> 32, `GROUP=1`** (1 day + one pod session). The main event.
3. Marlin-shaped **SR (register-B) decode kernel** in TileLang — proven
   expressible by the verdict above, gated on step 2's ncu. Copy
   `example_dequant_gemm_w4a8.py` (register-resident), not the
   `_bf16_fp4_hopper` WGMMA donor tileRL adapted. Lands as a per-arch sm90
   cell; Metal keeps the SS kernel.
4. sm90 bf16-IO kernel cell (`embedding`/`rmsnorm`/`rope`/`silu_mul` are still
   the CPU cell's f32 kernels on sm90). ~2.3 ms/tick and 4.7 GiB.
5. Fold `.oscale` into the kernel accumulator.

Honest expectation: **69–84 tok/s, central ~75 (0.89x Arle)**. Not 84.5, and it
should not be promised. The estimate comes from a tick model calibrated against
this repo's own numbers — it reproduces the published 19.02 ms / 52.6 tok/s
baseline to two decimals before being asked to predict anything.

## Kill criterion

One pod session, ~2 hours. Sweep `micro_size_k` in {8,16,32} x `GROUP` in {1,2,4}
on `gate_up` (34816x5120) and `down` (5120x17408) at M=1, direct kernel call,
same process. **If the best configuration does not exceed 1.30 TB/s on
down_proj** — Marlin's own measured rate on that exact shape
(`agent-infer/docs/experience/wins/2026-08-23-marlin-nvfp4-decode-bps-tiebreaker.md`)
— the memory-level-parallelism thesis is dead, step 2 is worth nothing, and
the fallback is no longer "vendor Marlin": the register-resident verdict makes
a TileLang-native SR kernel the fallback, behind the same sm90 registry cell.

Pair it with one ncu run reporting registers/thread and blocks/SM: if micro=32
spills to local memory, the repo has already seen that failure mode take a
kernel to 22% of roof (`wins/2026-08-25-fp4-gemv-grouped-dequant.md`).

## Judge split

Two of three lenses (long-term cost, minimality) ranked "widen the load, keep one
source" first. The evidence lens ranked **vendoring Marlin** first, on the
grounds that its payoff is arithmetic over four *measured* wall times while step
2's payoff rests on one integer whose rejection nobody wrote a number down for.
That disagreement is exactly what the kill criterion above resolves, for two
hours of GPU. The verdict above also cuts against the evidence lens's
vendoring-first rank: its premise was that step 3's payoff was unproven, and
step 3 is now proven reachable in-tree.
