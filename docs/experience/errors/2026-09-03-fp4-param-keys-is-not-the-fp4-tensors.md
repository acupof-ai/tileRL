# Two sessions derived the tick's byte count from a config list, not the checkpoint — 2026-09-03

## Context

`README.md` claimed a decode tick reads 22.8 GB, from
`docs/experience/2026-08-28-vs-sglang-h20.md:22`, which records it with no
derivation. Two sessions independently tried to rebuild it and both landed near
19.2 GB, three and a half short. One of them (this one) went as far as editing
the README to mark the roofline percentage "under revision" and publishing 53%.

**The recorded 22.8 was right.** Measured off `load_hf`'s params on the real
checkpoint: 24.44 GB resident, 21.89 GB streamed once the `embed_tokens` gather
is excluded — 4% from the recorded figure, and KV traffic at d512 covers that.

## Root Cause

`fp4_param_keys(cfg)` is derived from the config. It names the 497 keys that
*would* be fp4-packed when `cfg.fp4`; it never opens the checkpoint. On this
checkpoint only **264 carry an fp4 `.scale`** — the other **233 carry `.w8` /
`.wscale`, fp8 e4m3, one byte per element**, and `load_hf` decides which by
dispatching on what the file actually holds (`.weight_packed`, `.wq`,
`.weight_scale_inv`, `.weight_scale_2` — `model.py:697-731`).

Both derivations costed all 497 at 4 bits plus a scale. 10.6 GB of fp8 weights
were charged at roughly half their real width, which is the whole shortfall.

| | measured |
|---|---:|
| `w8` (float8_e4m3fn) | 10.625 GB |
| `wq` (uint8, fp4 nibbles) | 7.499 GB |
| `scale` (float32) | 3.746 GB |
| `embed_tokens` | 2.543 GB |
| `oscale`, `conv1d`, `wscale` | 0.024 GB |
| **resident** | **24.44 GB** |
| **streamed** (less the embed gather) | **21.89 GB** |

21.89 GB in 11.0 ms is 1.99 TB/s, **60%** of the H20's measured 3312 GB/s. The
README's 64% was approximately right; the 53% this session briefly published was
built on the same bad population and is retracted.

## Fix

README quotes 60% against the measured 21.89 GB, and names the population trap
next to it so the next reader does not repeat the derivation. The
`docs/experience/2026-08-28-vs-sglang-h20.md` entry still shows no derivation for
its own figure — it was right, and it was unauditable, and those are separate
problems.

The surviving lever, at its real size: the block scales are 3.746 GB resident as
f32 because `model.py:141` calls `renorm_fp4_scale(weight_scale.float(), ...)`
and `kernels_linear.py:546` declares `Scale` f32. As e4m3 they are 0.937 GB, so
2.81 GB — 13% of the tick, not the 25% derived from the bad population.

That change is **not free, and the argument that it was is also wrong.**
`renorm_fp4_scale` divides by `p = exp2(floor(log2(amax)))`, which moves the
exponent and preserves whatever mantissa it was handed — but `_native_fp4` hands
it `weight_scale.float()` with a global scale already folded in, and that product
is not on the e4m3 grid even though both factors were. Round-tripping all 936.6M
scale elements: **0 flush to zero** (smallest scale over largest row max is
4.687e-03 against e4m3's 1.953e-03 subnormal floor, 2.4× margin) but **0.0734%
change value**, concentrated in `in_proj_a`/`in_proj_b`. Each scale governs 16
fp4 weights, so ~1.2% of weights move by one e4m3 ulp. The discriminator is
MMLU 1000 with the scales round-tripped against the same slice unmodified.

## Rule

A key list named for a format describes intent, not bytes. Before costing a
tensor population, read the population off the loaded model — `param_specs` and
`fp4_param_keys` answer "what does the config ask for", and the question was
"what is on the disk". Both sessions made this error within an hour of one of
them naming the same error class in the other's work.
