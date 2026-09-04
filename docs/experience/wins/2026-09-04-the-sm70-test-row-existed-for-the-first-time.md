# The sm70 test row existed for the first time, and all 14 failures were the tests

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), against cpu / metal / sm90
**Commit:** 4b70614 measured; 1 remaining failure fixed in d0b438c

## Context

The compatibility question is isolation: one tree, every target runs it, per-target
tuning must not interfere across targets. Answering it needs a per-target test row.
sm70 had none — **pytest was never installed on that pod**, so the sm70 column was
not merely empty, it had never existed.

## Numbers

Full suite, `/work/tilerl-git` via `venv70` (`--system-site-packages`, tilelang
0.1.13 + torch 2.5.1+cu121), clean checkout verified by `git status --porcelain`:

| run | commit | result |
|---|---|---|
| first ever | b20ca9b | **13 failed**, 230 passed, 13 skipped |
| after 12 fixes | 4b70614 | **1 failed**, 243 passed, 14 skipped (160 s) |
| the last one | d0b438c | fixed; rerun pending pod idle |

Comparison row, same period:

```
cpu     9b2f182   251 passed,   8 skipped
metal   b20ca9b   244 passed,   4 failed  (same 4 on main -- clean)
sm90    9b2f182   244 passed,   4 failed
sm70    4b70614   243 passed,   1 failed
```

## Attribution

**14 test defects, 0 product defects.** Four classes, and every one is a test
encoding an assumption about the target:

| n | class | the assumption |
|---:|---|---|
| 8 | `write_tokens_f32 dtype` | the test's own `PagedKvPool` took bf16 by default against sm70's f32 kernel ABI |
| 3 | fp8 `equal_cpu` | `torch.equal` on `Float8_e4m3fn`, absent in torch 2.5.1 |
| 2 | fp4 parity M=16 | `target.startswith("cuda")` standing in for "has a w4a8 kernel" — true on sm90, false on sm70 |
| 1 | `65 == 64` | pad=0, i.e. no captured decode tick; `_graph_on` is True on CUDA and the padding row owns a block |

The fp4 pair is the interesting one: M=6 passed and M=16 failed with the boundary
at exactly `_MX = 8`, the reference's own fork point. The w4a8 reference quantizes
both operands to e4m3 (~2% error that does not average down over K), which is
where `max abs diff 6.695e-01` came from — the kernel was right and the reference
was the wrong one.

The `65 == 64` was found by the sm90 peer, not by me, and it fires on **both** CUDA
arches while passing on cpu and metal for the same reason: cpu/metal get pad=0. It
read an intentional reservation as an off-by-one in `_fit_blocks`.

## What it decided

The isolation evidence arrived as a side effect. **No failure was a product symbol
computing a different value per target.** The product tree already ran on three
targets; what did not was the test tree's picture of them. So the merge gate is
satisfied on the sm70 side, and the four remaining sm90 failures are the peer's
control to attribute — two of them (`test_linear_fp4_parity`,
`test_linear_fp4_gemv_parity`) are the fp4 twin of the fp8 `_pad2d` bug and are
fixed in d0b438c, so its control will not clear them: `main` passes those tests
while cropping a 64-row `wq` to 32.

Scope not covered: the sm70 row is a test-suite row, not a throughput row. It says
the kernels agree with their references on that card, not what they cost there.

## Rule

A target with no test row is not a target that passes — it is a target with no
evidence. Install the runner before claiming portability, and expect the first row
to be mostly test defects: assumptions that were free to make while only one
target ever ran.
