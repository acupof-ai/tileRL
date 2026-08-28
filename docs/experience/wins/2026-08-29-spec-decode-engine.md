# Speculative decode in the engine — H20, 2026-08-29

> Status: pending-remote

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

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src TILERL_TARGET=cuda \
      python3 scripts/bench_batch_decode.py /data00/Qwen3.8-27B-NVFP4 \
        --layers 64 --batches 1,8 --draft /data00/Qwen3.8-27B-NVFP4/model_mtp.safetensors \
        --depth 4

The `tok/tick` and `accept` columns are the verdict; `--draft` omitted is the
baseline arm.

## Rule

Pending. Nothing about spec-decode goodput on the 27B is settled until the
command above runs on GPU 7.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| | | | | | | | |

Raw artifacts: pending-remote.
