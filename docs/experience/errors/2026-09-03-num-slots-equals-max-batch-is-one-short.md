# num_slots == max_batch is one slot short — the pad row takes one, V100 sm70, 2026-09-03

> Status: **fixed, and it cost two 10-minute pod runs.** A decode tick with fewer rows than
> its graph bucket permanently reserves one state slot for padding rows, out of the same
> pool requests allocate from. With `num_slots == max_batch` that leaves `max_batch - 1` for
> requests and the next `submit()` raises `LinearStatePool exhausted`. The engine hides its
> own version of the failure, so only the caller ever sees it.

## What happens

`engine.py:827`, in `_run_decode_graph`:

```python
if n < B and self._pad_slot is None:
    try:
        self._pad_slot = self._states.alloc_slot()
        self._pad_block = self._kv.alloc_block()
    except RuntimeError:
        B = n  # no spare capacity to park padding rows on: exact size
```

One slot, taken on the first padded tick, never returned. And `n < B` is not exotic — with
`max_batch=4` the graph buckets are `(1, 2, 4, 8, ...)`, so:

| rows `n` | bucket `B` | pads? |
|---:|---:|:--:|
| 1 | 1 | no |
| 2 | 2 | no |
| **3** | **4** | **yes** |
| 4 | 4 | no |

`n=3` is guaranteed in a benchmark that drains a batch: requests retire one at a time, so
every window ends with ticks at 3, 2, 1 rows. The 3-row tick takes the slot.

## Why it took two runs

**The engine catches its own failure.** When `alloc_slot` raises inside that `try`, the
engine falls back to an exact-size graph and continues — correct behaviour for serving, and
it means the pool can be one slot short with no symptom until *someone else* allocates. The
someone else was my next `measure()` call, two functions away, and the traceback points at
`submit()` with no mention of padding.

**And the CPU target cannot reproduce it.** `_run_decode_graph` only runs where CUDA graphs
are captured, so on this GPU-less box `_pad_slot` stays `None` forever. I built a local
repro of the crash, ran it at `num_slots=4` and `num_slots=6`, and **both passed** — the
repro could not reach the path. That is the third failure this session that the Mac hid
(the first was `torch.cuda.synchronize()` blocking the repro outright).

What actually identified it was not a run but reading `_graph_bucket` and the pool: 12 lines
that say `n=3` pads, plus the knowledge that my own window logic — close at the first
completion, then drain — manufactures `n=3` ticks by construction. **The fix I shipped
between the two crashes created the trigger for the second one.**

## Fix

`num_slots = max_batch + 2` in the bench, with the reason at the assignment. Two rather
than one so a second padded bucket cannot repeat this.

The guard is a test that needs no GPU, because the arithmetic is pure bookkeeping:
`tests/test_e2e.py::test_a_padded_decode_tick_needs_a_spare_state_slot` asserts `n=3` pads
into the 4 bucket (so the test is guarding something), that a full batch does not, and that
`bench_ctx_decode.py` still sizes its pool above `max_batch`. **Negative control verified**:
the previous sizing fails both source assertions.

## Rule

**A caught exception moves the symptom, it does not remove it.** The engine's `except
RuntimeError` is right for serving and turns a resource shortage into a silent capacity
loss that surfaces in unrelated code. When a failure points at an allocation, check who
holds the resource, not who asked for it last.

Second: **a repro that passes both ways is not a repro.** I ran mine at the broken and the
fixed setting, got two passes, and treated the fix as verified. Two passes means the
experiment does not touch the mechanism — the same shape as the A/B whose arms agreed to
0.5%, one entry earlier, on the same day.

Third: **price a pod run before relaunching on a hypothesis.** Each attempt was ~10 minutes
of GPU plus the wait; the reading that settled it took under a minute and was available
before either launch.

## Results table

| date | commit | machine | target | model | measurement | value |
|---|---|---|---|---|---|---|
| 2026-09-03 | f755cae | V100 | cuda sm70 | qwen38-27b | rows that pad at `max_batch=4` | **n=3 → bucket 4** |
| 2026-09-03 | f755cae | V100 | cuda sm70 | qwen38-27b | slots held with `num_slots=4` | 3 requests + 1 pad → **next submit raises** |
| 2026-09-03 | f755cae | V100 | cuda sm70 | qwen38-27b | pod runs lost to this | **2 (~20 min)** |
| 2026-09-03 | f755cae | V100 | cuda sm70 | qwen38-27b | local repro at broken/fixed sizing | **both passed — CPU never captures a graph** |
