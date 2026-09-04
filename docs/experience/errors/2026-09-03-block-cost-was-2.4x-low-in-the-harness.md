# A KV block costs 2.125 MiB on sm70, not the 0.92 MB the harness printed, 2026-09-03

> Status: **corrected, pinned by a test.** `bench_ctx_decode.py` sized and printed its block
> pool at **0.92 MB/block**. Measured from the shipped pool classes it is **2.125 MiB
> (2.228 MB) — 2.42× more**. Nothing failed because of it; every "how much headroom did that
> arm have" figure derived from the printed pool size was simply wrong, and the server I
> started for the demo reserved **4.56 GB where I told the user 1.9 GB**.

## The measurement

`scripts/probe_block_bytes.py` builds the two pools the way `build_engine` does
(`engine.py:1191` for the trunk, `:358` for the draft's mirrored plane) and divides by
`num_blocks`:

| IO dtype | trunk | draft plane | per block |
|---|---:|---:|---:|
| **f32 (sm70)** | 2.0000 MiB | 0.1250 MiB | **2.1250 MiB = 2.2282 MB** |
| bf16 | 1.0000 MiB | 0.0625 MiB | 1.0625 MiB = 1.1141 MB |

The shape: 16 full-attn planes × 4 kv heads × 16 tokens × 256 head_dim × 4 B, for k and v.
sm70's attention IO is f32 and the pool matches it deliberately (`engine.py:1202` — a bf16
pool made every attention call cast the whole plane, 14% of a 4096-ctx token).

**So 0.92 MB was two errors at once**: it is a bf16 figure, *and* it omits the draft plane.
0.79 + 0.13 in the old comment is close to the bf16 trunk (1.00) plus bf16 draft (0.0625) —
so it was probably measured before the pool moved to f32 IO and never revisited.

## What it cost

Nothing broke. That is the point — a wrong constant here is invisible:

- **The pool print lies by 2.42×.** It was added specifically so two runs reporting a
  "ctx=512" row could be compared for free memory (`errors/2026-09-03-expandable-segments-is-load-bearing.md`
  records a run stalling because of exactly that). It now reports MiB, correctly.
- **The demo server over-reserved 3.9 GB.** I passed `--blocks 2048` believing it was 1.9 GB;
  it is **4.56 GB**. One request at ctx=4096 plus a 512-token completion needs **289 blocks
  (644 MB)** — the pool was **7× larger than the workload**. It fit inside 28.7 GB, so it
  produced no symptom, only a smaller headroom than I reported.
- A second stale constant in the same comment: "ctx 4096 at B=4 needs 1060" now computes
  **1092**, because `--tokens` moved from 32 to 128. Replaced with the formula rather than
  another number that will rot.

## Pinned

`tests/test_kv.py::test_a_block_costs_2125_kib_at_the_27b_shape` builds both pools at the real
27B config and asserts 2.125 MiB. **Negative control run**: set to 0.92 and it fails with the
measured value in the message.

The gate is worth its four lines only because the failure mode is silence — the arithmetic it
guards is in a comment and a `print`, neither of which any other test reads.

## Rule

**A constant in a comment is a claim with no gate on it.** This one was wrong for as long as
the pool has been f32 and nothing noticed, because the number is used to *describe* a run, not
to make it work. When a magic number appears in a print or a sizing comment, either derive it
in code or pin it in a test; a number that only humans read is a number that only humans can
check, and they don't.

Second, narrower: **`0.79 + 0.13` looked like it had been measured.** A decomposed constant
reads as more trustworthy than a round one, which is exactly why a stale decomposition survives
longer. The decomposition was right about the *structure* (trunk + draft plane) and wrong about
the dtype — the most durable kind of error, since checking the structure confirms it.

## Gate

194 passed, 4 skipped, ruff clean. Negative control on the new test verified. No GPU used —
the pool shapes are the same on the CPU target, only the dtype differs, and the test pins f32
explicitly rather than reading the local backend's.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | block cost, f32 IO (sm70) | **2.1250 MiB / 2.2282 MB** |
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | block cost, bf16 IO | 1.0625 MiB / 1.1141 MB |
| 2026-09-03 | (this) | Mac | cpu | qwen38-27b cfg | what the harness printed | **0.92 MB (2.42× low)** |
| 2026-09-03 | (this) | Mac | — | — | `--blocks 2048` as served | **4.56 GB** (reported as 1.9) |
| 2026-09-03 | (this) | Mac | — | — | blocks a 4096+512 request needs | **289 (644 MB)** |
| 2026-09-03 | (this) | Mac | — | — | stale "B=4 ctx=4096 needs 1060" | **1092** |
