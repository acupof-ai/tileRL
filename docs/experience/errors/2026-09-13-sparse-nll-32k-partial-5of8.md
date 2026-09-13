# 32k sparse-vs-dense teacher-forced NLL is partial at 5 of 8 windows — sm70 V100, 2026-09-13

> Status: partial (windows 3–7 of 8; project wrapped up before windows 0–2 ran)

## Context

The sparse long-context fidelity record at 32k on sm70 (V100, f32 KV) used
`fidelity_engine.py --nll`: 64-token greedy continuations of a dense engine
scored under each arm's model on prefix-aligned disjoint 32768-token windows
of one held zh-wiki stream. Eight windows (w0–w7) were planned; only w3–w7
completed before the project moved off its own runtime. The runs are
forced-teacher, B=1, sm70 — they do not predict free-running sm90 generation.

## What Worked

Per window, sparse-minus-dense mean NLL gap (nats/token) and sparse top-5
agreement:

| window | k=128 gap | k=256 gap | k=128 top5 |
|---|---:|---:|---:|
| w3 | −0.3255 | −0.0547 | 1.0000 |
| w4 | +0.0053 | −0.0978 | 1.0000 |
| w5 | −0.1995 | −0.0286 | 1.0000 |
| w6 | −0.0708 | +0.0171 | 1.0000 |
| w7 | −0.1103 | −0.1999 | 0.9688 |

n=5 mean: k=128 gap **−0.1402**, k=256 gap **−0.0728**; top-5 agreement
**0.9938** (k=128) and **1.0000** (k=256). Negative gaps mean the sparse
continuation scored lower NLL under the sparse model than under dense — a
calibration/selection signature, not an error; per-window spread is wide
(−0.33 to +0.005 at k=128), which is why the 8-window mean was the planned
verdict statistic and the partial n=5 mean is not a substitute.

Two failed w0–w2 attempts are recorded in `nll012.log`: the first OOM'd
against another session's 256k capacity probe; the retry used a driver missing
the nvcc-12.4 PATH bootstrap and failed kernel compilation (`nvcc 11.8: c++20
not defined`). Both were launch/config faults, not measurement failures; the
runs were killed unwound at project wrap-up.

## Rule

A negative sparse-minus-dense teacher-forced NLL gap with ~1.0 top-5
agreement is a selection/calibration observation, not sparse correctness — and
a mean over only 5 of 8 planned windows stays Status: partial.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-12/13 | 9efd3833 | V100 sm70 | 32k nll k=128/256 n=5/8 | Qwen3.8-27B-NVFP4 | — | — | — |

Raw artifacts: V100 `/home/chenkailun.c/tilerl-logs/fidelity-nll-32k-w{3,4,5,6,7}.json`,
`nll67.log`, `nll012.log` (failed w0–2 attempts).
