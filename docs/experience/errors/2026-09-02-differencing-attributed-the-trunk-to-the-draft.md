# Differencing attributed the trunk's cost to the draft — V100 (sm70), 2026-09-02

> Status: corrected. The withdrawn claim was "one draft forward adds 955 torch
> elementwise launches, each further depth step adds 29 — 33× first-vs-later".
> The draft's real share is **8 launches, 0.02 ms**. The 955 were the trunk's.

## Context

A depth-3 spec tick spends ~9 ms across ~1400 torch elementwise/index kernels,
17% of GPU-busy and none of it TileLang work. `with_stack=True` returns no Python
frames on this build (torch 2.5.1+cu121), so attribution had to come from
somewhere else. I differenced configurations: count a `dense` tick, a `spec d1`
tick, a `spec d3` tick, and read `spec d1 - dense` as "one draft forward".

That difference read 955 launches, and `(d3 - d1)/2` read 29 — a 33× ratio that
one 1-layer draft head cannot produce. I recorded it as unexplained and refused
to name a cause, which was right, but I kept the number.

## Root Cause

**`spec d1` and `dense` do not differ only by a draft forward.** A verify tick
runs W trunk rows where a dense tick runs 1, and the trunk's GDN layers take a
completely different code path at W>1:

```python
if q.shape[1] == 1 and not getattr(kv, "dense", False):
    out = backend.gdn_decode(...)      # T=1: one fused launch, pool updated IN PLACE
if out is None:                        # T>1 (or off sm90): gather -> chunk -> scatter
    state, window = backend.state_gather(...)
    out, new_state, new_window = backend.linear_attn_chunk(...)
    backend.state_scatter(...)
```

So `spec d1 - dense` = one draft forward **+ 48 GDN layers switching from a fused
in-place kernel to a gather/scatter pair**. Everything on the right of that plus
sign landed in the column labelled "draft".

The difference was arithmetically correct and semantically wrong: two
configurations that differ in two things cannot attribute to one of them.

## Fix

Attribute directly instead of subtracting. `torch.profiler` nests CUDA events
under the CPU range that launched them, so wrapping the model's own methods in
`record_function` gives exact per-region counts — and `event.name` on the parent
is the **aten op**, which is the attribution `with_stack` could not supply:

```python
for meth, region in (("_full_attn", "trunk.attn"), ("_gdn", "trunk.gdn"),
                     ("_mlp", "trunk.mlp"), ("_linear", "trunk.linear")):
    _wrap(model_mod.Model, meth, region)
```

The profiler's serialization is what makes this exact rather than what breaks it:
launch counts and summed kernel durations survive it (only wall clock and host
share do not), and a serialized launch cannot be misattributed to a neighbouring
range.

`scripts/prof_region_attrib.py`.

## Results

Depth 3, prompt 30, one steady tick, `decode_graph=off` (regions need CPU ranges):

| region | launches | torch ms |
|---|---:|---:|
| **trunk.gdn** | **792** | **6.40** |
| — (bookkeeping between regions) | 58 | 0.54 |
| trunk.attn | 324 | 0.75 |
| trunk.linear | 183 | 0.60 |
| trunk.mlp | 138 | 0.51 |
| **draft.fwd + draft.conf** | **8** | **0.02** |
| TOTAL | 1502 | 8.80 |

Per aten op, and against the dense negative control:

| region / aten op | verify d3 | dense |
|---|---|---|
| `trunk.gdn/aten::copy_` | 528 / 2.68 ms | 48 / 0.10 ms |
| `trunk.gdn/aten::_index_put_impl_` | 96 / **2.45 ms** | **0** |
| `trunk.gdn/aten::index` | 144 / **1.13 ms** | **0** |
| trunk.gdn total | 792 / 6.40 ms | 58 / **0.12 ms** |

240 launches / 3.57 ms is exactly `state_gather`'s 3 fancy-index gathers plus
`state_scatter`'s 2 index_puts (reference.py:857-882), × 48 GDN layers. The
dense control has none of them because `gdn_decode_fused` updates the pool in
place — which is also the negative control for the whole attribution: if the
regions were mislabelled, the two runs would not differ in exactly the ops the
source says differ.

That traffic is ~302 MB/tick, 0.34 ms at 900 GB/s, so it runs **10.5× off its own
byte cost** — indexed-elementwise kernels doing index arithmetic per element, not
bandwidth.

The withdrawn numbers, for the record: "one draft forward 955 launches / 6.91 ms,
each further 29" — replaced by 8 launches / 0.02 ms for the whole draft.

## Rule

**A difference between two configurations attributes to one thing only if they
differ in one thing.** Write down what else changed before reading the subtraction
— here a verify tick's W>1 flips 48 layers onto a different branch, which is
visible in the source at the `if q.shape[1] == 1` two lines from where the cost is.

Second: **when the profiler cannot give you a stack, it can still give you the
parent range — put the ranges in yourself.** Six `record_function` wrappers
located in one run what two differencing passes could not, and they carry the
aten op name for free.

Third, the one I got right and should keep: I refused to publish a cause for the
33× ratio while it was unexplained ("没查清就不报因果"). The ratio was an artifact
of my own instrument. Publishing the mechanism would have been publishing fiction
about the draft path.

## Gate

None — this is a measurement method, not runtime code. `prof_region_attrib.py`
carries its own control: run it with `--depth 0` and the index ops must vanish.
