# sm70 27B decoded id 220 forever — it was layer truncation, not a kernel bug — 2026-08-30

> Context: bringing up the V100 (sm70) fp4 path. The full 64-layer model was
> too slow in eager to iterate on (549s for 4 tokens), so correctness was
> checked on `load_hf(num_layers=N)` truncations. Every truncation (1, 4, 8
> layers) decoded id 220 (whitespace) forever. Hours went into hunting a
> "hidden-state collapse" in the fp4 GEMV, oscale, GDN, attention, rope.

## Root cause

**A layer-truncated model degenerating to the highest-frequency token is
normal, not a bug.** A 1–8 layer slice of a 64-layer model has no semantic
depth; its argmax collapses to the corpus-most-frequent token, which for this
tokenizer is id 220 (space). The tell was there and misread: the logprob was
**−4.1, not the −12.4 uniform floor** — the model was *confidently* choosing
space, exactly what a shallow model does, not a numerically collapsed one. The
full 64-layer model decodes correctly and confidently: "The capital of France
is" → " Paris.\nThe", logprobs −0.30 to −1.40.

## What actually threw the hours off

The premise "220 = bug" was never checked against a truncated *correct* model
before the hunt began. The elimination chain (tiny cpu==sm70 identical across
fp4, fused, asymmetric GDN, partial rope, GDN-first) all passed — correctly,
because the sm70 path *was* right — but each pass was read as "the bug is
elsewhere" instead of "there is no bug; the test is wrong." The measurement was
wrong, not the thing measured (the third time this exact failure mode bit the
project this week — see the peer's lm_head-sharding and the take()-pops-finished
harness bug).

## Three real bugs found on the way (all genuine, all shipped fixes)

These were real and are the actual sm70 bring-up content — the "CUDA == has
bf16" class the peer flagged in the original handoff:
1. `_rows` cast activations to bf16 for every `target.startswith("cuda")`, but
   sm70's cell is the f32 CPU kernel set → bf16 into an f32 kernel. Fixed with
   `Backend.io` (bf16 only when `arch=="sm90"`, else f32).
2. `embedding` had the same `target.startswith("cuda")` gate → bf16 embed table
   fed an f32 downstream. Gated on `self.io` instead.
3. `linear_fp4`'s M>1 generic fallback fed bf16 `x2` to the f32
   `make_linear_fp4` (sm70 is the first cuda cell with no fp4 MMA kernel, so the
   first to reach that path). Fixed with `self._f32(x2)`.

## Rule

Before treating a degenerate decode as a kernel bug, run the SAME truncation
against a target known-good (CPU, or a smaller trusted model) — a shallow model
is *supposed* to output the highest-frequency token. And read the logprob: a
confident wrong token (logprob ≫ uniform floor) is a shallow/mis-prompted
model, NOT a collapsed one; a collapsed model sits near −log(vocab). Validate
correctness on the FULL model or a config the reference also degenerates on —
never on a truncation whose degeneration you have not first characterized.

## Meta-rule: a green signal proves a narrower thing than it reads as

This bug and three the peer hit the same week are one failure mode — reading a
narrow guarantee as a broad one. Before trusting a green/zero signal, ask what
it actually constrains:
- **CUDA graph capture succeeds** ⟹ the *captured region* has no host sync. It
  does NOT ⟹ the tick is sync-free: the sampler's token read-back runs after
  replay, outside capture, and must (decode tick = 0 scalar + 1 bulk, not 0/0).
- **`aten._local_scalar_dense` count == N** ⟹ N scalar `int()`/`.item()`. It
  does NOT ⟹ N transfers: multi-element `.tolist()`/`.numpy()` are invisible to
  a TorchDispatchMode counter (`scripts/probe_syncs.py` wraps both classes).
- **A parity test is green** ⟹ the kernel faithfully implements *its reference's
  algorithm*. It does NOT ⟹ the algorithm is accurate: an fp8-activation
  reference and its kernel agree at 3.6e-2 while the project's 1e-2 gate is a
  different bar (see the M=4..8 fp8-decode path).
- **A truncated model degenerates** ⟹ it is shallow. It does NOT ⟹ a bug.
When the signal is green, name the exact claim it licenses before building on it.

