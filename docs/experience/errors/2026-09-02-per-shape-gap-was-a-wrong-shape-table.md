# The sm70 per-shape "5-33% of peak" gap was a wrong shape table — 2026-09-02

> Status: Task #21 closed as **not a defect**. In the captured graph the fp4
> GEMV runs at **746 GB/s = 83% of peak**. The per-shape microbench that said
> 5-33% was measuring shapes the model does not launch.

## Context

`scripts/ab_scale_f16.py` prints achieved GB/s per fp4 shape at M=1 and reported
a strongly non-uniform gap: `attn o` (N=1024) at **5% of peak**, `gdn out` and
`gdn z` at 32-33%, weighted whole-token 526 GB/s = 58%. That produced a task with
a stated diagnosis (n_partition=4 cannot fill 80 SMs at small N) and two levers
(raise n_partition, fuse the small shapes).

The task itself carried the right next step, and it is the one that saved the
work: **re-measure in the captured graph before optimizing**, because the
microbench has a ~60 µs eager launch floor the graph path does not pay.

## Root Cause

In the graph, the GEMV is 19.35 ms of a 27.70 ms token and streams 14.43 GB —
**746 GB/s, 83% of the 900 GB/s peak**. There is 3.32 ms of headroom in the whole
GEMV class, 12% of a token, not the 42% the microbench implied.

The microbench's shape table was wrong in three places. It is written as
UNFUSED projections, but serving runs `fuse_projections`, and two rows are simply
not the checkpoint's shapes:

| bench row | bench (N×K, ×count) | actually launched |
|---|---|---|
| `mlp gate/up` | 17408×5120 ×128 | 34816×5120 ×64 (fused `gate_up`) |
| `gdn qkv` + `gdn z` | 10240×5120 ×48, 6144×5120 ×48 | 16384×5120 ×48 (fused `qkvz`) |
| `gdn out` | 5120×6144 ×**64** | 5120×6144 ×**48** — GDN is 48 layers, not 64 |
| `attn o` | **1024**×5120 ×**32** | **5120×6144** ×**16** — `o_proj` is (h, hq·d) = (5120, 6144) |
| `gdn ab` | absent | 96×5120 ×48 |

`attn o` is the row that produced "5% of peak", and N=1024 appears nowhere in
this model: the spec is `specs[f"{p}.o_proj"] = (h, hq * d)`, and with hidden
5120 / 24 heads / head_dim 256 that is 5120×6144. The 1024 looks like an
`hq*d` computed with a head_dim of 64 rather than 256, and the count 32 double-
counts a per-layer projection that exists 16 times.

**Why it survived review**: the script asserts its own total —
`assert abs(nib_tot/1e9 - 12.81) < 0.02` — and the table PASSES it at 12.799 GB.
The per-row errors cancel: `attn o` is 168 MB short and `gdn out` is 252 MB long.
A total-only check cannot see a redistribution, and every per-shape conclusion
drawn from the table was about the distribution.

The launch count is the check that would have caught it instantly. The table
implies 401 GEMV launches per token; the profiler measures **305**, which is
exactly what the fused shapes predict. That agreement is also what makes the
83% trustworthy: two independent derivations of the same 305.

## Fix

`SHAPES` rewritten to the fused shapes actually launched, and the assert
strengthened from one total to three: nibble bytes, **launch count**, and shape
arity. A table that reproduces the byte total but not the launch count is still
wrong, and only the launch count distinguishes fused from unfused.

Task #21's two proposed levers are dropped. Both were priced against a 42% gap
that does not exist:

- **Raise `n_partition`** — the shape it was aimed at (N=1024) is not launched.
  The smallest real shape is `gdn ab` at N=96, and it is 11.8 MB/token, 0.1% of
  the stream. Nothing to win.
- **Fuse the small shapes** — `gdn out`/`gdn z` were the target. `gdn z` is
  ALREADY fused into `qkvz`; the table listed the pre-fusion members.

## What the graph profile does say

One real item, found in the same profile rather than by hypothesis:

```
                       linear_fp4_gemv_sm70_m_kernel    19.35      305.0
adWithCast<1>, at::native::memory::StoreWithCast<1>)     1.64      305.0
```

**305 f32→f16 casts of X, 1.64 ms/token, 5.9% of the token.** One per GEMV
launch, from `backend.py:554` (`.to(torch.float16)` inside the chunk loop). The
bytes are 14.5 MB/token = 0.016 ms at peak, so the measured 1.64 ms is **102×
the byte cost**: it is 305 kernel launches, not traffic. Filed as its own task —
it is a launch-count problem with a real number attached, which is what #21 was
supposed to be and wasn't.

## Rule

**A benchmark's shape table is code, and a total-only assert does not test it.**
When per-shape conclusions are the output, assert the per-shape structure —
count and arity, not just the sum. Errors that redistribute bytes between rows
pass a total check by construction, and the distribution was the whole finding.

Corollary, and the reason this cost one profile instead of a week of kernel work:
**derive the launch count and compare it to the profiler before believing any
per-shape number.** 401 against 305 is a 31% discrepancy in the most basic
observable, available before any timing.

Third: a shape table hand-copied from a model's parameter list drifts the moment
serving fuses anything. `_projection_groups` is the single source of truth for
what gets fused, and a bench that enumerates shapes should read it rather than
restate it.
