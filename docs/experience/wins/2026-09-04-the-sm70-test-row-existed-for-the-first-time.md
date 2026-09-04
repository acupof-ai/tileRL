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

## sm90 (H20), 200 GSM8K, greedy, `max_new_tokens=512`, decode graph on

Both arms in one process on card 7, one job on the host. Branch at `9b2f182`,
baseline at `40bc83c`, each from a `git worktree` checkout rather than a file
push — the pod tree used for the baseline was verified against the sha by
enumerating all 154 `.py` files, not the changeset (`src/` and `packages/`:
zero differences).

**The draft path did not move.** At B=1 the branch accepts **54329 of 73983**
drafted tokens — byte-identical counts to the baseline run. At B=8 it is
53867/72828 against 53911/72947, which is 6.18 accepted per block of 8 on
both; the totals differ because the completions differ by one question. That
is a stronger statement than any wall-clock ratio below, and it is the one
that says the batched selector walk and the sm70 work leave the draft
arithmetic alone.

| cell | baseline | 9b2f182 | ratio | GSM8K | tok/decode-fwd |
|---|---:|---:|---:|---|---:|
| B=1 base | 78.4 | 82.0 | 1.046 | 168/200 → 168/200 | 1.00 |
| B=1 spec w8 | 135.5 | 135.8 | 1.002 | 167/200 → 167/200 | 6.12 |
| B=8 base | 242.9 | 245.9 | 1.012 | 165/200 → 165/200 | 7.79 |
| B=8 spec w8 | 225.4 | 230.6 | 1.023 | 163/200 → 164/200 | 42.55 |

**No cell regressed, and none is outside run-to-run spread.** The gate is
0.97x and the worst cell is 1.002x. **No direction is claimed on the deltas**:
there is one draw per cell, `bench_ctx_decode`'s measured spread is 0.3–0.9%
on tok/s, and three of the four deltas fall inside 2.3%. The B=1 base cell at
+4.6% is the only one outside that band and it has no second draw.

B=1 speculation still beats its own base arm by **1.656x** (135.8 / 82.0), so
the accept verdict for speculative decode at B=1 survives the branch.

**Two things this section does not establish.** Throughput parity is not a
correctness gate — the `_tl_layout` rename, whose stale reader built an empty
exemption set and failed only on sm90 with a served forward, would have run at
full speed. And `_fit_blocks`' CUDA arithmetic is not exercised here:
`acc_spec_arms.py` passes `num_blocks=512` explicitly, so `num_blocks=0` is
never taken, and `_fit_rows` cannot differ from `_pad2d` at these shapes even
in principle — `bN > 32` is required and the 27B's N are all multiples of 128.
That is why an N=24 parity test found the re-target bug and no throughput run
ever could.

*(This section is the sm90 owner's own text, landed unedited.)*


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

Scope not covered, beyond the two limits the sm90 section states for itself: the
sm70 row is a test-suite row, not a throughput row — it says the kernels agree with
their references on that card, not what they cost there. `serve --blocks 0` on a
real card is the one open item at the time of writing (#62's other half).



## Rule

A target with no test row is not a target that passes — it is a target with no
evidence. Install the runner before claiming portability, and expect the first row
to be mostly test defects: assumptions that were free to make while only one
target ever ran.
