# In-place GDN state: read-after-write on one global buffer ×8.7 slower — cuda(H20), 2026-08-28

## Context

Moved the fused GDN decode kernel's state I/O in place into the pool
(`States[Slots[b], layer]`) to kill the per-layer gather/scatter index kernels.
Correctness PASS; B=1 tick regressed 71.3 → 63.4 tok/s.

## Root Cause

In-graph profile: `gdn_decode_fused` 6.5 → **56.8 µs/call**. The two j-loops
loaded and stored the SAME global buffer element by element; with the old
separate `NewState` buffer the loads had no dependency on the stores. Aliased
load-after-store on global memory serializes every iteration on the previous
store's completion (~K=128 round trips per pass).

## Fix

Stage the decayed column in registers (`s_loc[K]`): pass 1 loads + decays,
pass 2 updates from registers and stores once. Back to 6.8 µs/call; the
gather/scatter kernels are gone.

## Rule

Never read and write the same global buffer inside a per-element loop in a
kernel — stage through registers/shared and write once. After any kernel
change, price it in-graph (`profile_graph_kernels.py`) before trusting the
tick; the harness only says "slower", the profile says which kernel.
