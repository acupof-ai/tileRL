# PD Disaggregation & AFD — Design Note

Design only. Both are deployment topologies over the existing seams — no
engine or model surgery.

Reference: Step-3 AFD (arXiv 2507.19427), SGLang PD disaggregation docs.

## Why the current design fits

| Requirement | tileRL seam today |
|---|---|
| KV transfer unit | `PagedKvPool` blocks (16 tokens, refcounted, prefix-shared) + block tables |
| GDN state transfer unit | `LinearStatePool` slots, already a separate store from KV |
| Layer-typed split | `Model.forward` iterates layers by type (`full_attn_layers` vs GDN) |
| Role-agnostic runtime | `Engine` submit/poll + `StepLimits` — same class drives any role |
| Quantized transport | fp4/fp8 ops + the (precision, arch) kernel registry |

## PD disaggregation

- P engine: prefill only, writes KV blocks + GDN state slots.
- D engine: decode only, reads them.
- Handoff: block table + KV blocks + state slot, transferred once per request
  at prefill completion. Prefix cache stays on P; D is stateless across requests.
- Missing (additive): a KV transport (NCCL/RDMA), a router that sends
  prefill to P and decode to D, and D-side admission that waits for handoff.

## AFD (Attention–FFN disaggregation)

Step-3 split, mapped onto the hybrid model:

- **Attention pool**: full-attention sublayers + embedding + lm_head. Owns the
  paged KV cache. DP-replicated (per-token state lives here). Small batches.
- **FFN pool**: GDN linear-attention sublayers + all MLPs. Owns the GDN
  recurrent states (`LinearStatePool`). Large batches (compute-bound).
- No KV crosses the network — the key Step-3 property, and already true here
  because the two state stores are separate.
- Transport: hidden states at layer boundaries (A→F quantized, F→A bf16 per
  Step-3), overlapped with a 3-stage micro-batch pipeline.
- Missing (additive): activation transport, a split executor that pipelines
  A/F micro-batches, and per-pool kernel selection via the registry (A on
  bandwidth-optimized hardware, F on compute-optimized).

## Order

1. PD first — one transfer per request, engine seam already role-agnostic.
2. AFD after — per-layer transport and pipelining are a bigger build; only
   worth it once a single engine is batch-limited.
