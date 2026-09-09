# Speculative decode in the engine — H20, 2026-08-29

> Status: Shipped — the 27B goodput question is answered as of 2026-09-09; see
> [`2026-09-09-spec-decode-h20-b1-loses-on-the-serving-build.md`](2026-09-09-spec-decode-h20-b1-loses-on-the-serving-build.md).
> On the H20 serving build, B=1 spec is a net loss at every depth (depth 1: 0.986x).

## Context

The engine now drafts and verifies inside its normal tick: a decode row drafts
up to `spec_depth` tokens off the trunk's last hidden, and the same forward
verifies them as a `seq_q = 1+depth` row. The number that decides whether it
ships is committed tokens per tick against the tick's cost — not acceptance
rate on its own.

Two things the design turns on, both verified on the pod earlier this week
(`scripts/parity_chunk_vs_decode.py`): a mid-sequence multi-token forward
agrees with T=1 greedy decode exactly, and the trunk's paged KV needs no
rollback (a rejected draft's slot is overwritten by the next tick).

The gated-delta recurrent state DOES need a rewind, and it is taken without a
second forward: the GDN chunk path writes the state and conv window after every
chain step (`BatchKv.keep_steps` -> `LinearStatePool.step_states`), and the
engine adopts the plane at the accepted length. Cost is 151 MB x depth of extra
writes per row per tick (~1.4% of an 11 ms tick) and ~+4.8 GB of pool at B=8,
depth 4 — against a whole extra forward for the snapshot/re-absorb alternative.
Non-spec ticks pay nothing but one KS=1 store into a reused scratch pair (the
fused kernel needs the operands; nobody reads them).

## What Worked

Verified on CPU only so far: with a draft attached the engine emits
token-for-token what it emits without one, both when every draft is rejected
(random head), when every draft is accepted (oracle head), and when the policy
trims every chain below `spec_depth` —
`tests/test_e2e.py::test_speculation_reproduces_greedy_decode`. Deliberately
mis-indexing either half of the rewind (state or conv window) fails that gate,
and so does writing the step planes at full pool width instead of the tick's.

Unmeasured here: acceptance rate and ms/tick on the 27B. No GPU on this host.
Measured 2026-09-09 on H20 card 6 — see the entry linked at the top.

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src TILERL_TARGET=cuda \
      python3 scripts/bench_batch_decode.py /data00/Qwen3.8-27B-NVFP4 \
        --layers 64 --batches 1,8 --draft /data00/Qwen3.8-27B-NVFP4/model_mtp.safetensors \
        --depth 4

The `tok/tick` and `accept` columns are the verdict; `--draft` omitted is the
baseline arm.

## Rule

Pending. Nothing about spec-decode goodput on the 27B is settled until the
command above runs on GPU 7.

Settled 2026-09-09 on H20 card 6: on the serving build, B=1 spec loses at every
depth (depth 1: 0.986x baseline). The machinery works; the H20's rung prices do
not pay for it at B=1. Full table in the 2026-09-09 entry linked at the top.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-09 | 53d349a | H20 card 6 | cuda/sm90 fused+graph | Qwen3.8-27B-NVFP4 | — | 10.569 baseline / 20.718 d1 | 94.6 / 93.3 (B=1) |

Raw artifacts: `/work/specg2{base,d1,d2,d3}.log` on the pod; full table in the
2026-09-09 entry.
