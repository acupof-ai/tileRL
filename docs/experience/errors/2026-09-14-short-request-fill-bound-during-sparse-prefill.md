# A short request during a long sparse prefill runs at 5.59 tok/s — 2026-09-14

> Status: open. Measured on V100 sm70 during the #586 hybrid deploy run.

## Context

Hybrid serve (#586, `--sparse-min-tokens`): a short dense request served
while a unique 128k sparse prompt fills measured **5.59 tok/s, 3.2 s TTFT**
vs ~52 tok/s solo. Sparse ticks cost ~1 s median on sm70. The earlier 32k
measurement was 9.9 tok/s / 2.5 s TTFT; the 128k fill is the worse point.

## Root cause

Device-queue blocking, not scheduling: rolling-window wall-time fairness is
correct and slots are free, but a dense decode tick syncs behind the
in-flight sparse prefill kernel on the device. A prefill cap of 96 shortens
the waits but raises total prefill time by 124% (195.8 → 438.0 s on the 32k
fill), so 192 stays the default.

## Fix

None. Upgrade path is a sparse prefill kernel or chunking work that lets
dense decode pass the fill rather than queueing behind it.

## Rule

Fair scheduling across two paths is not tail-latency isolation when the
paths share one device queue; the bound is the slow path's tick length.
