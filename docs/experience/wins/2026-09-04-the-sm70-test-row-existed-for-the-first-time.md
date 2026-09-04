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
| the last one | d0b438c | fixed |
| **green** | **ff75987** | **0 failed, 245 passed, 14 skipped (51.9 s)** |

One run in between reported 7 failed / 7 errors and was **not** a code result: the
V100's root filesystem was **100% full, 0 bytes available**, so every test writing
under `/tmp` died in `safetensors` with `No space left on device` or in pytest's own
`tmp_path` factory. Nothing was deleted to fix it — `/tmp/hf_cache` alone is 37 GB
of someone else's model cache — the suite runner now sets `TMPDIR` to
`/data00/home/chenkailun.c/pytmp` on the 46 GB-free volume. Worth recording because
14 simultaneous failures across two unrelated files reads exactly like a code
regression, and the discriminator was one `df -h`.

Comparison row, same period:

```
cpu     9b2f182   251 passed,   8 skipped
metal   b20ca9b   244 passed,   4 failed  (same 4 on main -- clean)
sm90    d0b438c   248 passed,   1 failed  (main: 200 passed, 1 failed, SAME test)
sm70    ff75987   245 passed,   0 failed
```

sm90 throughput, 200 GSM8K greedy at `max_new_tokens=512`, decode graph on, one
process per arm, against the 40bc83c baseline (the peer's cells): B=1 base
78.4 → 82.0, B=1 spec 135.5 → 135.8, B=8 base 242.9 → 245.9, B=8 spec 225.4 → 230.6.
**Worst ratio 1.002x against a 0.97x gate, no cell regressed.** The +1.0 to +4.6%
is not claimed as a win: one draw per cell, and `bench_ctx_decode`'s measured
spread is 0.3–0.9%, so three of four deltas are inside noise. Acceptance is the
evidence that the draft path did not move — B=1 is **54329/73983 accepted,
byte-identical to main**; B=8 is 53867/72828 against 53911/72947, 6.18 per block of
8 on both.

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
targets; what did not was the test tree's picture of them.

**The matrix is complete and every failure in it is attributed.** metal's 4 are
pre-existing, confirmed by running `main` under metal. sm90's four on 9b2f182
resolve as: two fp4 `_pad2d` (pre-existing reliance on `F.pad`'s crop — `main`
passes them while cropping a 64-row `wq` to 32), one new test with a wrong
expectation (the pad row; product sizing byte-identical on both), and
`[chunked]`, which **fails on `main` too** and is now explained as a drift/margin
lottery, not a draft bug
([errors/2026-09-04-the-argmax-gate-was-a-lottery-on-the-drift.md](2026-09-04-the-argmax-gate-was-a-lottery-on-the-drift.md)).
sm70's 14 across the whole run: all test defects, 0 product defects.

Scope not covered: the sm70 row is a test-suite row, not a throughput row — it says
the kernels agree with their references on that card, not what they cost there.
Throughput parity is not a correctness gate either, which the `_tl_layout` bug
already demonstrated. And `_fit_blocks`' CUDA arithmetic is **still unexercised on
any card**: `test_serve_sizes_...` turned out to be a test defect rather than the
coverage it looked like, so `serve --blocks 0` remains unmeasured (#62's open half).


## Rule

A target with no test row is not a target that passes — it is a target with no
evidence. Install the runner before claiming portability, and expect the first row
to be mostly test defects: assumptions that were free to make while only one
target ever ran.
