# `s[j] *= x` on a register array lowers wrong in tilelang 0.1.13 — cuda(H20), 2026-09-03

## Context

Generalizing `make_gdn_decode_fused` from one token per row to T, the
recurrence was rewritten to update the register-resident state column in
place, because the state now has to survive across the T steps:

```python
# WRONG — no error, no warning, 70% wrong state
for j in T.serial(K):
    s_loc[j] *= exp_g_s[0]
    kv_mem[0] += s_loc[j] * k_s[j]
delta[0] = (v_s[tv] - kv_mem[0]) * beta_s[0]
for j in T.serial(K):
    s_loc[j] += delta[0] * k_s[j]
    acc_o[0] += s_loc[j] * q_s[j]
```

`s_loc` is `T.alloc_local((K,), "float32")`. Every kernel compiled, every
shape matched, nothing raised. The state came out 1.685e-02 off a reference
whose own maximum is 2.424e-02 — 70% relative — at **T=1**, the width the
kernel already handled correctly before the rewrite.

## Root Cause

Not the T-loop. The same arithmetic written as a let binding followed by an
explicit store, which is what `make_gdn_chunk_fused` has always done, is
exact:

```python
# RIGHT — 3.3e-09 against reference.gdn_forward
for j in T.serial(K // 4):
    base = j * 4
    for u in T.unroll(4):
        sj = s_loc[base + u] * exp_g_s[0]
        s_loc[base + u] = sj
        accs[u] += sj * k_s[base + u]
```

The two forms are the same arithmetic on paper. tilelang 0.1.13 lowers them
differently, and the in-place form reads back something other than what it
stored. Augmented assignment is not the trigger on its own — `accs[u] += ...`
in the correct form is one too, and it is fine. The two forms also differ in
their loop structure (`T.serial(K)` vs `T.unroll(4)` inside `T.serial(K/4)`),
and nothing here isolates which of the two differences is the trigger. That is
a question for a tilelang reproducer, not for this kernel.

## Fix

Copy `make_gdn_chunk_fused`'s recurrence verbatim into
`make_gdn_decode_fused`: let-then-store, four accumulators, `T.unroll(4)`
inside a `T.serial(K // 4)`. Decode kernel vs `reference.gdn_forward`, tiny
shapes (B=2, nvh=4, K=V=16), f32:

| | T=1 | T=4 |
|---|---:|---:|
| final state | 3.3e-09 | 2.8e-09 |
| per-chain-step states | 3.3e-09 | 7.2e-09 |
| conv window, step windows | 0.0 | 0.0 |
| out (bf16 output cast) | 3.8e-04 | 2.7e-04 |

Tighter than `gdn_chunk_fused` itself (3.1e-05 state), which is bf16-IO.

## Rule

Copy the chunk kernel's recurrence verbatim; do not paraphrase it into an
in-place update. When a rewritten kernel is wrong, diff its **T=1** path
against the untouched kernel line by line before hunting in the code you
added — T=1 is the case the old kernel already got right, so any T=1 error
is in the rewrite, not in the generalization. Component-wise diffs against
`reference.gdn_forward` (state, step states, window, step windows, out
separately) localize it in one pod round trip: here the windows came back
exact and the state did not, which ruled out every index expression and
pointed at the arithmetic.
