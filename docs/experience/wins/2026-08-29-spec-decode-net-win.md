# Speculative decoding — REJECTED: measured against the eager path, not the shipped one

> Status: Rejected. The engine keeps the code (it is correct and gated); the
> feature does not pay until a speculative tick can be graph-captured.

## Context

Speculation landed in the engine: a decode row drafts off the trunk's last
hidden and the same forward verifies it as a `seq_q = 1+depth` row. Goodput is
real — **1.87 committed tokens per trunk forward at depth 2, 43-47% acceptance**
— and this entry first reported it as a 1.14-1.19x win.

That was wrong. The comparison was against the EAGER decode path, and tileRL
ships a CUDA-graph decode. `Engine.__init__` disables graph capture whenever a
draft is present (`self._decode_graph_on = decode_graph and draft is None`),
because a captured graph replays one T=1 step through the fused `gdn_decode`
kernel, which does not exist at T>1.

Measured, same script, same session (H20, GPU 7, 64 layers, 15 timed ticks):

| arm | B=1 ms/tick | B=1 tok/s | B=8 ms/tick | B=8 aggregate |
|---|---:|---:|---:|---:|
| eager, no draft | 91.6 | 10.9 | 138.1 | 57.9 |
| **graph, no draft (shipped)** | **11.6** | **86.2** | **60.0** | **133.3** |
| speculation, depth 2 (forced eager) | 106.2 | 17.6 | 151.0 | 92.3 |

**Against what ships, speculation is 4.9x slower at B=1 and 1.44x slower at
B=8.** The graph is worth 7.9x on its own at B=1 (785 launches collapsed into
one replay); 1.87 tokens per forward does not buy that back.

## What Was Actually Established

- The draft head works: teacher-forced top-1 agreement 84.4%, 43-47%
  acceptance in the loop, 1.87 tok/tick at depth 2.
- The verify path is correct — a mid-sequence multi-token forward matches T=1
  greedy exactly, and the block loop reproduces greedy with an always-wrong and
  an always-right draft.
- Serving the head as fp8 is worth ~75x per projection against the generic
  dense path (9.7 ms -> 0.13 ms for the same shape).

None of that is retracted. Only the throughput verdict is.

## The Real Work

Capture a speculative tick: a graph family per `(B, 1+depth)`. `_DecodeGraph`
today has static `[B,1]` id/position buffers, a static all-ones `seq_q_lens`
deliberately outside the captured region, and the fused `gdn_decode` path. A
spec tick needs `[B,1+depth]` buffers and the chunk GDN path — a different
graph, not the same one reshaped.

## Rule

Benchmark against the configuration that ships, not the one that is easy to
instrument. The eager path was the convenient baseline because
`bench_batch_decode` defaults to it; the number it produced was real and the
conclusion drawn from it was false. Two rows in one baseline file, side by
side, is what exposed it — which is the argument for one snapshot covering
every path rather than a per-feature script.
