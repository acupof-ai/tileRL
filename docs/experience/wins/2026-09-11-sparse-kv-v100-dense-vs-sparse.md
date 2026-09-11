# Sparse KV day 2: Quest dense-vs-sparse attention GPU time on the V100 — sm70, 2026-09-11

> Status: **GPU-time curve measured on the V100 sm70, 27B NVFP4, 65536-token
> pool** (`scripts/bench_sparse_gpu_v100.py`). Sparse attention is flat
> 0.37–0.43 ms per call from 4k to 64k; dense reaches 3.46 ms at 64k — **9.3x**
> on the attention kernel there. The wall-clock control
> (`scripts/bench_sparse_v100.py`) is kept because it produced a finding about
> instrumentation: graphs-off decode is host-dispatch-bound, so wall time
> cannot see the saving.

## Context

Day 1 landed the CPU selector and the sm70 scorer cells
([2026-09-11-sparse-kv-page-bounds-cpu.md](2026-09-11-sparse-kv-page-bounds-cpu.md)).
The card half asks whether selecting k=128 pages + the forced 8-page V4.1
window actually buys decode time on the 27B at 8k/32k/64k, every page
device-resident.

## The serve that occupied the V100

The V100 was held by ckl's own restarting serve: supervisor `bash
scripts/serve_v100.sh` pid **3171691** (started Mon Sep 7 23:01:56 2026,
elapsed 3d13h, cwd `/data00/home/chenkailun.c/tilerl-git`, log
`/data00/home/chenkailun.c/serve70c.log`) and its worker pid **349099**
(`venv70/bin/python -m tilerl.cli serve --model qwen38-27b --draft
…model-00018-of-00018.safetensors --depth 1 --port 8000 --max-batch 1
--max-ctx 32768`, started Sep 11 09:52:16, 25.9 GiB, zero established :8000
connections). Under ckl's clearance ("你直接用 您自己的服务可以随便搞", relayed by
a3) the supervisor was killed first (TERM, so nothing respawned) and the
worker exited with its parent; no -9 needed. nvidia-smi read **0 MiB** and no
new worker appeared within 62 s. Killing the worker alone would have let the
restart loop (up to 10) fight the bench for the card.

## Wall time is the wrong instrument with graphs off

The selection wrapper is Python around `backend.paged_attention`, so both arms
run with decode graphs OFF in one process
(`scripts/bench_sparse_v100.py`, f32 KV pool, 64 measured decode tokens,
median of 2 draws):

| ctx | dense tok/s | sparse tok/s | ratio | index ms/tok (GPU events) |
|---:|---:|---:|---:|---:|
| 8192 | 8.4 | 7.6 | 0.90x | 1.28 |
| 32768 | 8.0 | 7.5 | 0.93x | 2.07 |

Dense staying at ~8 tok/s across a 4x context growth is the tell: graph-on
serving historically scales 38.0 tok/s at 8k to 15.3 at 32k
([2026-09-04-decode-reaches-32k-and-the-tick-slope-grows.md](2026-09-04-decode-reaches-32k-and-the-tick-slope-grows.md)),
so attention DOES grow there — eager graphs-off hides it under host dispatch
and the per-tick D2H syncs (`int(seq_lens[0])` on every layer, the
`write_tokens` tolist path). The sparse arm additionally pays the host-side
selection (GPU topk + scatter then `.numel()`), and its wall loss (~12 ms at
8k) dwarfs the 1.28 ms the scorer costs ON the GPU. Wall tok/s answers
"eager + Python selector vs eager", not "sparse attention vs dense
attention".

## GPU-time sweep (the controlled measurement)

`scripts/bench_sparse_gpu_v100.py`: one prefill to 65536, then on the first
decode tick the populated pool is swept at histories
512/1024/2048/4096/8192/16384/32768/65536. The unchanged sm70 split
paged_attention is called on the first L/16 pages (dense) and on the scorer's
128+8 selection among them, both timed with CUDA events (median of 10 after
3 warms; per-table-width kernel recompiles land in the warms). Results:

## Results

| history L | dense pages | sparse pages (128+8 union) | dense attn ms | sparse attn ms | attn ratio | scorer ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 32 | 32 | 0.370 | 0.372 | 0.99x | 0.139 |
| 1024 | 64 | 64 | 0.391 | 0.425 | 0.92x | 0.158 |
| 2048 | 128 | 128 | 0.439 | 0.429 | 1.02x | 0.139 |
| 4096 | 256 | 130 | 0.419 | 0.398 | 1.05x | 0.143 |
| 8192 | 512 | 135 | 0.650 | 0.394 | 1.65x | 0.196 |
| 16384 | 1024 | 136 | 0.958 | 0.427 | 2.24x | 0.238 |
| 32768 | 2048 | 136 | 1.706 | 0.370 | 4.61x | 0.358 |
| 65536 | 4096 | 128 | 3.458 | 0.372 | 9.30x | 0.549 |

Median of 10 CUDA-event calls after 3 warms, one source plane (the 4 planes of
a group dispatch identically). Below k+window dense and sparse read the same
pages, so the ratio hovers at 1.0 — selection only removes work once the
history exceeds the selected set (~4k). From there dense scales linearly with
history (the expected bandwidth-bound KV read: 3.46 ms / 4096 pages =
0.84 us/page), while sparse stays at the 128-page floor. The scorer grows with
the candidate page count because it touches every page's f16 bound, but at
0.55 ms at 65536 it is one sixth of the 3.09 ms attention delta on that call.

Scaled to a decode token (16 full-attn layers, score computed on the 4 group
sources): at 64k dense attention is ~55 ms/token (16 x 3.46), sparse attention
~6 ms (16 x 0.37) plus ~2.2 ms scoring (4 x 0.55) — roughly 47 ms/token of GPU
time returned, before counting that weight GEMV is the rest of the tick. The
token/s claim has to run with the graph ON (the eager host path below erases
it); that is the H20 arm through `build_engine(sparse_k=128)`, not this probe.

## Rule

A wall-time A/B whose control arm is dispatch-bound measures dispatch twice.
When the treatment changes only GPU work, time the GPU work with events on a
populated pool — one long prefill buys the whole history curve. Sparse
attention crosses at roughly the selected-set size: there is no saving while
dense already reads <= k+window pages, then the ratio grows with history.

## Results log

| date | machine | target | result |
|---|---|---|---|
| 2026-09-11 | V100-SXM2-32GB | cuda sm70 | wall 8k 0.90x / 32k 0.93x (dispatch-bound, graphs off); attn GPU-ms 0.99x@512 to 9.30x@65536, scorer <=0.55 ms |
