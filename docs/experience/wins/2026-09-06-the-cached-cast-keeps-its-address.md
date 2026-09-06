# The cached cast moved, so the graph baked a dead address — cpu, 2026-09-06

> Status: **Fixed on cpu; the payoff is `pending-remote`.** The address stability
> is gated here. Whether it lets an RL step skip its recapture is a card-6
> measurement that has not run — the number to beat is **34.09 s/step**
> (`wins/2026-09-05-recapture-after-update.md`).

## Context

`grpo_loop` calls `invalidate_weights()` after every optimizer step, which drops
every captured decode graph; the next tick that needs one re-captures it. The
marker at `engine.py:234` says why:

```python
# ponytail: no recapture after training — the graph bakes the f32 embed cast.
```

The question was whether that is necessary. A captured CUDA graph bakes the
**address** of every tensor its kernels read, so it survives a weight update
exactly when the update writes through the same addresses.

## Root Cause

**The optimizer already does.** `autograd.py`, `AdamW.step_one` ends:

```python
p.copy_(p32.to(p.dtype))      # in place; p.data_ptr() unchanged
```

So the parameter is not what moves. `backend.py`'s `_const_f32` is:

```python
key = (t.data_ptr(), pad_to, dtype)
...
elif ver == t._version:
    return c
c = self._dev(t, dtype)        # NEW allocation, NEW address
```

`copy_` bumps `t._version`, the cache misses, and the converted copy is
re-materialised somewhere else. **The cast was the only thing moving**, and it is
not just the embedding the marker names — `_const_f32` has **27 call sites**:
the embedding table (`:1467`, and only when the table dtype differs from
`embed_io`), rmsnorm weights (`:514`), the fp4 `oscale`/`wscale` sites, and the
gdn `dt_bias` / `a_log` / `conv1d_weight` / `norm_weight`.

**LoRA does not pass through here.** `add_lora` builds `lora_a`/`lora_b` at
`precision.dtype("adapter")` = bf16 (`model.py:552`), and `model.py:238` sends
them through `backend.linear`, which casts with `_dev` (`backend.py:582`) — no
cache, no version key, no baked address. So for a LoRA run the cast targets above
are the whole story, and the adapters themselves are already stable.

## Fix

Keep the stale buffer when the parameter is the same object and only `_version`
moved, then refill it:

```python
else:
    stale = c  # same parameter, new values
...
if stale is not None and stale.shape == c.shape and stale.dtype == c.dtype:
    c = stale.copy_(c)
```

The shape/dtype guard is not decoration: `pad_to` and `dtype` are part of the
cache key, and a refill into a buffer of the wrong shape would be a silent
corruption rather than a miss.

## Gate

`tests/test_precision.py::test_the_cached_cast_keeps_its_address_across_an_optimizer_step`,
four arms — address held across a `copy_`, values equal to a fresh cast, and one
arm each for the two guards, because an untested guard arm is the one production
takes.

The value arm is the one that matters most: address stability with the *old*
values is a worse bug than the one being fixed, and both look identical from a
`data_ptr()` check alone.

Negative control, with `c = stale.copy_(c)` removed:

```
>       assert second.data_ptr() == addr, (
E       AssertionError: the cached cast moved across an optimizer step: a captured
E       graph baked 0x75ad2b480 and would replay stale bytes
E       assert 31588529408 == 31588529280
```

Only that arm goes red; the two guard arms stay green, which is what says they
are testing the guard and not the refill.

**One arm of the test was wrong before the code was.** The first version padded a
2-D tensor and asserted `shape[0] == 12`; it failed at 8. `pad_to` compares
`shape[0]` but `F.pad(c, (0, n))` fills the **last** dim — the two agree only for
a vector, which is what every real call site passes (a per-row scale). The test
now uses a 1-D tensor and says so. A red arm is not automatically the code's.

## Rule

**A cache that a captured graph reads must be keyed on identity and refilled in
place, not rebound.** Address stability is the contract; recomputing the value is
fine, moving it is not. And when checking such a cache, assert the values as well
as the address — a buffer that holds its address and its stale contents passes
every pointer check and fails every replay.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | (this) | Mac M-series | cpu | tiny | — | — | — (address gate, not a timing) |
| pending-remote | | H20 card 6 | cuda sm90 | Qwen3.8-27B-NVFP4 | | | recapture s/step vs 34.09 |

**What the remote run must show**, beyond a faster step: a graph captured before
an optimizer step, replayed after it, must equal an eager forward on the new
weights. A speed number for a cache with no staleness check beside it is
"faster" and "wrong" at once — the same control
`wins/2026-09-05-recapture-after-update.md` had to add.
