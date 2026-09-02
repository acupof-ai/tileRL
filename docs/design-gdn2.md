# GDN2 — Adoption Assessment

Not implemented; no GDN2 model in scope (Qwen3.8-27B uses GDN1).

Source: NVlabs/GatedDeltaNet-2 (arXiv:2605.22791), "Decoupling Erase and Write".

## What changes vs GDN1

```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
```

- `b_t` — channel-wise erase gate (key axis); `w_t` — channel-wise write
  gate (value axis); `D_t = Diag(α_t)` — channel-wise decay.
- Strict generalization: scalar gates → GDN1. The erase gate is the main
  quality driver (RULER multi-key retrieval ~28 → 37.8).

## Can we copy it?

- **Code: no.** NVIDIA Source Code License-NC (non-commercial) — and the
  kernels are Triton, which doesn't fit the tilelang-only backend anyway.
- **Algorithm: yes.** The recurrence is a paper equation, not code. Clean-room
  tilelang implementation is legitimate and is a small delta over the GDN1
  kernels: two extra gate projections (`in_proj_b`/`in_proj_w` naming TBD by
  the first GDN2 checkpoint), channel-wise decay, chunkwise WY forward +
  gate-aware backward.
- tilelang ecosystem has no GDN2 kernels yet — we'd be the first.

## Trigger to implement

A GDN2 checkpoint appears in the target family (check `layer_types` /
`linear_attn` config for erase+write gates). Until then, YAGNI.
