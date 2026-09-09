# A mutant that never ran

2026-09-10. `tests/test_e2e.py::test_the_fp8_kv_pool_generates_what_the_bf16_pool_does`.

## Context

The test proves the fp8 KV pool generates the same tokens as the bf16 pool. Token
agreement alone cannot see the defect the flag exists for — a scale-less store
generates the same 6 tokens on this fixture — so the test also carries a mutant:
shift `k_scale` and assert the output *changes*, proving the gate reads the scale
plane. The first version of that mutant monkeypatched `_store_fp8` to write a wrong
scale, then asserted the fp8 arm's output changed.

On CUDA the assertion passed with the output unchanged. The conclusion drawn was
"the CUDA kernel doesn't read `k_scale`".

## Root cause

`_store_fp8` is a CPU-path seam. The CUDA store never calls it. A call counter on
the patched function settled it: **calls = 0** — the mutation never executed, so
"output unchanged" proved nothing about the kernel. The reading ruled out the
conclusion it had been taken to support: with zero calls, the result is identical
whether or not the kernel reads the scale.

The first conclusion looked true because the assertion's shape — "mutate, then
assert the output changed" — has no guard that the mutation ran. A dead mutant and
a correct kernel produce the same observation.

## Fix

Mutate the data, not the path. After the first step (the 40-token prompt prefills
in one tick, so the scales are written and every decode reads them after), roll the
scale plane in place:

```python
before = pool.k_scale.clone()
pool.k_scale.copy_(pool.k_scale.roll(1, 1))  # dim 1 = blocks
assert not torch.equal(before, pool.k_scale), ...
```

Rolling the plane itself is backend-agnostic — it asks only whether reads look at
`k_scale`, and it cannot be dead on one backend and alive on another. Verified on
pod CUDA: 6/6 tokens changed. The `calls` counter was the diagnostic that found
the root cause; the permanent guard is the before/after inequality, which proves
the mutation changed state on every backend that runs the test.

## Rule

1. An assertion of the form "the mutation did not change the result" must be
   preceded by an assertion that the mutation executed and changed state. Without
   it, a dead mutant and a correct system are the same observation, and the test
   passes green on a mutation that never happened.

2. A write-path seam cannot prove a read-path property where the two paths fork by
   backend. The patch runs on the CPU store; the claim is about the CUDA read.
   Mutate the data both paths touch, not one path's seam.

## Same-shape scan, 2026-09-10

Criterion: a test patches a seam and asserts on the result; if the patch is never
called on some backend, does the test go green (this disease) or red? Every
candidate in `tests/` that patches a seam to assert on behavior:

| test | patch | uncalled → | verdict |
|---|---|---|---|
| `test_e2e.py::test_load_kv_writes_the_same_bytes_in_two_calls_not_two_per_block` | wraps `torch.Tensor.copy_` with a `calls` counter, then `assert not calls` | green on the counter alone — but `assert torch.equal(got, ref)` against a nonzero ref fails if `load_kv` did nothing, so a dead patch cannot pass the whole test | safe, guarded by the equality arm |
| `test_decode_graph.py:357` | `e._graph_for = lambda B, W, keep: asked.append((B, W))` | `assert asked == [(4, 1)]` fails on `[]` | safe, self-proving |
| `test_decode_graph.py:551` | `e._graph_for` wrapper recording into `asked` | the count assertion fails on `[]` | safe, self-proving |
| `dp_world4.py` scramble control | `tr._order_agrees` made to disagree | no exception raised → `return 1` (red); also CPU-only (`RefBackend`), no backend fork | safe, self-proving |
| `tp_backend_world2.py:48` `no_collective` | `all_reduce` → identity | the probe is `rank+1` so a live collective disagrees with the control's expectation → red | safe, self-proving |
| `tp_world2.py:43` `no_fork` | `_tp_fork` → identity (drops the backward collective) | real fork runs → gradients match → the control expecting divergence fails → red | safe, self-proving |
| `tp_world2.py:148` `local_stats` | `shard_dim` → `None` | real sharding runs → stats match → control fails → red | safe, self-proving |
| `test_kv.py:165` `test_prefix_hash_collision_is_verified` | `store._roll = lambda *_: 0` forces every sequence to hash 0 | **green** — with real rolling hashes, different tokens hash differently, so every `lookup(... ) is None` and every same-sequence hit still passes; the test stops testing collision verification silently | **same disease, latent** — the patch is live today, but no assertion guards that it stays live; a one-line guard (`assert store._hash_all(toks) == store._hash_all(other)`) makes it self-proving. Fix in a separate PR |

One instance of the disease beyond the fp8 mutant: the hash-collision test. The rest are
self-proving — spies assert on the call record, negative controls assert the breakage is
visible, and the `load_kv` test's nonzero equality arm fails if `load_kv` did nothing.
