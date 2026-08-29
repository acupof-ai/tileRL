# Prefill: the last cheap lever is closed, and the gap is arithmetic — 2026-08-29

> Status: Closed. Prefill stays at 2237 tok/s against sglang's 4022 on the same
> batch, and the remaining distance is not tuning.

## Why this was re-measured

`gdn_chunk_core` — the torch-level chunkwise-WY path behind
`TILERL_GDN_CHUNKWISE` — was rewritten today to call the new `_gdn_chunk_fwd`,
the same helper the chunked backward uses. The earlier rejection of torch
chunkwise measured DIFFERENT code, so quoting it at the new code would have
been assuming rather than knowing.

Prefill suite, 27B, H20 GPU 7, three lengths each:

| GDN path | len 512 | len 2048 | len 8192 | vs fused |
|---|---:|---:|---:|---:|
| **fused kernel (shipped)** | **2237** | 2218 | 2143 | 1.00x |
| torch chunkwise C=16 | 493 | 489 | 496 | 0.22x |
| torch chunkwise C=32 | 841 | 838 | 834 | 0.38x |
| torch chunkwise C=64 | 1123 | 1085 | 1105 | 0.50x |

It loses 2-4.5x and improves monotonically with chunk size — the launch-bound
signature, since a bigger chunk is fewer launches for the same work. Even at
C=64 it is half the fused kernel. **The rejection holds for the rewritten code.**

(The same helper is a 1.63x WIN in the backward. Nothing contradictory: the
backward it replaced ran ~28 launches per TIME STEP, so chunking cut launches
there; the forward it competes with is already one fused kernel per layer.)

## The gap, as arithmetic

- Prefill is 913.9 ms where sglang's rate implies 509 ms.
- GDN is 27.6% of it. **Delete the GDN kernel entirely — set it to zero — and
  prefill reaches 3160 tok/s, still short of 4022.**
- So closing the gap needs a near-perfect linear-attention kernel AND about
  1.36x more from the GEMMs, which already run at 59% of fp8 peak.

Every cheap lever on the GDN kernel has now been measured and rejected:
chunkwise-WY (three times, two implementations), the V split (duplicates the
per-block q/k work), shared-memory state (2.5x slower — the old note was
right), the q/k prologue (2-3%, under the gate), the ptxas register level, and
the K split (its cost model is verified free; three attempts could not make
KSP>1 compile). The kernel is register-limited at 255 registers/thread with
6.25% occupancy and 4.5M local-load sectors per launch — it spills, and every
cheap fix moves which resource binds without moving the time.

## Rule

State a gap as arithmetic, not as a verdict. "Not SOTA" invites another round
of tuning; "zeroing the single largest kernel still leaves you 21% short"
names what would actually have to change — a different linear-attention
algorithm, or different hardware — and stops the search honestly.
