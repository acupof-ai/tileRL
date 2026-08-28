# Choose the KV split count by occupancy, not by pool depth — H20/sm90, 2026-08-28

> Status: Shipped

## Context

Split-KV decode attention picked 64 splits only when the pool reached past 64K
tokens, and 16 otherwise. That threshold was written to bound the serial scan
at long context. It says nothing about whether the GPU is full: the decode grid
is `(splits, kv_heads, batch)`, so at B=1 with 4 KV heads, 16 splits is 64
blocks on a 78-SM card — under half the machine, with the other half idle for
the whole attention.

## What Worked

Pick 64 whenever the 16-split grid would under-fill the card AND each split
still gets a whole page:

```python
wide = 16 * hkv * b < 2 * self._sms and max_tokens >= 64 * k_cache.shape[2]
ks, sfx = (64, "_64") if (max_tokens > 65536 or wide) else (16, "")
```

`self._sms` is read from the device, not hardcoded. B=8 keeps 16 splits (its
grid is already 512 blocks), so the change is B=1-only by construction — which
also makes it cleanly attributable in a mixed A/B.

Matched A/B, 27B NVFP4, GPU 7:

| row | before | after |
|---|---:|---:|
| decode-kv/d512-b1 | 90.9 | **92.4** (+1.7%) |
| decode-kv/d8192-b1 | 87.9 | **88.9** (+1.1%) |
| decode-kv/d32768-b1 | 78.6 | **80.1** (+1.9%) |
| decode-kv/d512-b8 | 308.6 | 308.0 (unchanged, as designed) |

## Rule

A split count is an occupancy knob before it is a scan-length knob. Size the
decode grid against the SM count of the card you are on — the same kernel
under-fills or saturates depending on batch and KV heads, and only the grid
arithmetic tells you which.

Corollary, learned the hard way here: the anomaly that prompted this (`d32768`
reading 28.6 tok/s, slower than `d131072` at 61.1) was a STALE baseline row
from before split-KV landed, not a live defect. Re-measure the row before
theorizing about it.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | — | 10.82 (B=1, d512) | B=1 **92.4 / 88.9 / 80.1** at 512/8k/32k |

Raw artifacts: `/work/chain3.log`; `docs/experience/wins/bench-baseline.json`.
