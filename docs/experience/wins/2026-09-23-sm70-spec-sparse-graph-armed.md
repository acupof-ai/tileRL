# Speculated sparse capture graph armed on sm70 — V100 sm70, 2026-09-23

> Status: Shipped (sm70 only; sm90 and every other CUDA arch stay guarded)

## Context

`_sparse_capture_allowed` (guard A, #805) kept the sparse captured decode graph
off on **every** CUDA arch whenever speculation was on (`draft` +
`spec_depth>=1`): the width-2 captured verify replayed trunk logits/hidden that
disagreed with the eager verify, so drafts stopped accepting and valid-but-wrong
in-vocab tokens appeared. That was correct for the build it was written against.

The `keep_steps=W` fix (PR #808) then changed what that verify replays.
`SparseDecodeGraph` had built its per-tick `BatchKv` with `keep_steps=int(W > 1)`
(1 at W=2) while eager passed `keep_steps=width` (2); `keep_steps=1` failed the
`gdn_decode_fused` eligibility gate (`t > 1 and keep_steps != t`) and fell back
to the GDN chunk kernel, casting q/k/v/z through bf16 where eager kept the fused
f32 path. With the fix, both paths build the same way.

This is a **default flip**, so it gets a bench entry: the fence is lifted only on
the arch where the fix was measured, and the arch list is the switch.

## What worked

`_SPEC_SPARSE_GRAPH_VERIFIED_ARCHS = ("sm70",)` in `src/tilerl/engine.py`.
`_sparse_capture_allowed` now returns, under speculation on CUDA, membership in
that list. Consequences, deliberately chosen:

- **sm70** arms the sparse capture with speculation on.
- **sm90 and any other CUDA arch** stay guarded — **unverified is not the same
  as fixed**, and sm90 has not been measured.
- **An unidentified arch fails safe**: the lookup is
  `getattr(backend, "arch", "") in <list>`, so a backend that does not carry
  `arch` is guarded rather than allowed by omission.
- `spec_depth=0` and non-CUDA behaviour are unchanged, as is the CPU cell
  (`CpuSparseGraph` is the token-exact W=2 oracle and stays enabled).

### Correctness — split in two, because the two are not the same evidence

1. **The fix itself — measured, already landed.** #808's device A/B on V100
   sm70, two trees differing only in the `keep_steps` lines: **6/6 cells MATCH**
   with `tok/fwd` strictly equal to eager, against 3/3 control W=2 cells
   `ALIGNMENT_UNMATCHED` (1.382 / 1.880 / 1.679). Both arms ran in both orders,
   no position effect.
   — [wins/2026-09-23-sparse-w2-graph-keep-steps.md](2026-09-23-sparse-w2-graph-keep-steps.md)
2. **The service shape at the new default — `pending-remote`.** Supplied by the
   production cutover window's full-sequence comparison (n=6 real 37.6k prompts
   + 1 below 8k + 1 near 16k, temp 0, both arms dumping per-prompt tokens,
   compared sequence by sequence). **Criterion: every sequence identical, or the
   first divergence at >= 128 generated tokens with the reason explained per
   prompt; failing either, no cutover.** This entry will be updated with that
   result in a follow-up docs PR; until then this half is open.

### Speed — what is measured, and what is not

The service window `servewin-0924-003135` measured **24.915 tok/s** effective on
the graph arm (6 real 37.6k wikitext prompts, warm window = ticks [16, end) with
the close tick excluded, `warm_effective_tokens` 5957 over 3270 ticks).

Two limits on how that number may be used, both from the window's own artifacts:

- **The window exited rc14** — `INSUFFICIENT: only 6 prompts (floor 20)`, with
  `probe.out` reading `{"ref_eager_w2048": 0, "graph_w2048": 14}`. The 24.915
  measurement stands on its own terms, but the window did not pass its own
  protocol, so it is a measurement and not a green verdict.
- **It has no same-window eager arm to compare against.** The probe set
  `measure = arm in ("graph_w2048", "baseline")`, so `ref_eager_w2048` wrote no
  throughput block at all. Any "vs the eager arm" figure from this window would
  be invented. The same-window speed ratio is **`pending-remote`** and comes
  from the cutover window, which now measures the eager arm too.

There is no correctness claim from this window either: its 09-17 first-replay
value gate fired **once** across the six prompts (prompt 0, bucket 4096,
`match=True`), and the 6/6 it does support is the *length* precondition
`len(ref) == len(output)` — a precondition for a value comparison, not a value
comparison.

## Rule

**A captured sparse decode graph may be armed under speculation only on an arch
where the width-2 verify has been measured correct against an eager reference;
the verified set is explicit and everything else stays guarded.** On sm70 the
fix is measured (6/6 cells) and the default flips there. The service-shape
correctness and the same-window speed ratio are measured separately in the
cutover window, because a repair's own A/B is not evidence about the shape it
is deployed in.
