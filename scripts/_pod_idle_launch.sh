#!/usr/bin/env bash
# Dev-only: launch the idle-window e4m3 bench — slice2/slice4 graph profile +
# roofline, three GPUs in parallel. JIT is cached from the contended run.
set -e
cd /work/tilerl

setsid bash -c 'CUDA_VISIBLE_DEVICES=1 TILERL_TARGET=cuda PYTHONPATH=src python3 scripts/profile_slice.py /host/tc27-nvfp4-slice2 --layers 2 --decode-graph --decode-ticks 30 > /work/idle_f32_slice2.log 2>&1; touch /work/idle_f32_slice2.done' </dev/null >/dev/null 2>&1 &
setsid bash -c 'CUDA_VISIBLE_DEVICES=2 TILERL_TARGET=cuda PYTHONPATH=src python3 scripts/profile_slice.py /host/tc27-nvfp4-slice4 --layers 4 --decode-graph --decode-ticks 30 > /work/idle_f32_slice4.log 2>&1; touch /work/idle_f32_slice4.done' </dev/null >/dev/null 2>&1 &
setsid bash -c 'CUDA_VISIBLE_DEVICES=3 TILERL_TARGET=cuda PYTHONPATH=src python3 scripts/bench_fp4_gemv.py /host/tc27-nvfp4-slice2 --layers 2 --ticks 10 > /work/idle_f32_gemv.log 2>&1; touch /work/idle_f32_gemv.done' </dev/null >/dev/null 2>&1 &
echo "LAUNCHED pids: $!"
