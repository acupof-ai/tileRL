# A pattern `pkill` orphaned an engine that held 73 GB of GPU 7 for 20 minutes — 2026-08-28

## Context

Three queued jobs (split-count A/B, 27B LoRA train row, sglang sanity check) all
died with `CUDA out of memory ... Process 2882081 has 72.00 GiB in use`. GPU 7
showed 73.8 GB used at 19% utilization with no process of ours visible: the PID
nvidia-smi prints is a HOST pid, and inside the container `ps -p 2882081` says
nothing. It read exactly like another tenant had taken the card.

## Root Cause

It was ours. Deconflicting a job queue, I ran `pkill -f "bash /work/sg.sh"`,
which killed the shell that owned a running engine and left the CUDA-holding
child alive with no parent — a repeat of a trap already recorded on this pod.
The other tenant's job (`torchrun --nproc_per_node=7 train.py`) sits on GPUs 0–6
and never touched 7.

## Fix

Find the holder by the fds it owns, not by nvidia-smi's host pid:

```sh
for p in $(ls /proc | grep -E '^[0-9]+$'); do
  ls -l /proc/$p/fd 2>/dev/null | grep -q nvidia && \
    echo "$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-90)"
done
```

The orphan was one line in that list (`bench_harness.py --suite train`, no
parent). `kill -9 <pid>` on that one pid took GPU 7 from 73.8 GB to 0.

## Rule

Never `pkill -f` on the pod — a pattern kills the shell and orphans the process
that actually holds the memory, and the orphan is invisible to `nvidia-smi`'s
pid column from inside a container. Enumerate holders by their `/dev/nvidia*`
fds, verify each cmdline is yours, kill by pid. And "another tenant took the
card" is a conclusion that needs that check first — twice now it was my own
leftover.
