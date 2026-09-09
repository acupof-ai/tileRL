# aupai-dd's external serve readings on H20 (2026-09-09)

## Context

aupai-dd ran our serve stack on H20 independently and authorized citation.
This is the first external reading of our serving stack. **External,
aupai-dd, 2026-09-09 — we have not reproduced it.** Marked the same way we
mark the four sglang numbers.

Config: `--max-ctx 4096 --max-batch 32 --slots 32`, `--blocks` not passed
(fitted: blocks_total=8192 = 131K tokens of KV).

| Reading | Value | Conditions |
|---|---|---|
| Single-stream decode | **88.0 tok/s** | B=1, warm, 512 tokens, card 4 |
| Warm aggregate | **695–726 tok/s** | 3 cards, 32 concurrent, B≈10.7/card → 232–242 tok/s/card |
| Cold (incl. JIT) | 249 tok/s | first-batch shapes pay compile |
| JIT warmup | 248 s one-time | "7 decode graphs in 248s" |

## What Worked

1. **88.0 vs our recorded 92.4 — a 5% gap, smaller than expected.** Our 92.4
   (`decode_tok_s`, README, harness decode-kv suite; store record id
   pending-remote) is a synthetic microbenchmark; theirs is under real HTTP
   load. An independent reproduction within 5% under a different load shape
   is the stronger evidence for the number.

2. **Cold 249 → warm 726 is 2.8x, and the store has no cell for it.** The
   serve-side JIT cost of the first batch shape — 248 s, one-time — has never
   been recorded on our side. A deployer must know it exists: a cold server
   looks 2.8x slower than the same server warm. This is a genuine gap in the
   registry, not just in this entry.

3. **The 249 they first reported was a false number, and they caught it
   themselves.** It folded the one-time 248 s warmup into the steady-state
   denominator. This is another instance of the cold/warm rule — the same one
   behind reporting CI's cold 6.4x and not the warm 14x: the two states answer
   different questions and must be reported separately.

Not in the store, deliberately: store records require a self-collected
commit sha, and an external reading cannot satisfy that. This is a real
store-boundary case — the entry is the record, the store stays pure.

## Rule

A throughput number must carry its warm state. Folding a one-time JIT cost
into the steady-state denominator makes a correct system look 2.8x slower —
and the error is invisible because both numbers are true measurements.
