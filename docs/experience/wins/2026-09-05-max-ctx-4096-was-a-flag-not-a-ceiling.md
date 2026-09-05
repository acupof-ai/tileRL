# `--max-ctx 4096` was a flag, not a ceiling — V100 sm70, 2026-09-05

> Status: Shipped (doc + flag). The restart is **pending-remote**: the pod's
> checkout sits on branch `docs-disk-full` at `550740a`, whose
> `serve_v100.sh:51` still reads 4096, so the flag does not take effect until the
> pod fetches this change.

## Context

ckl's report: "V100 上开的上下文长度有点太短了,输出到很短就截断了" — the served
context is too short, output truncates.

**The symptom was misattributed, and that changes the fix.** Output was not being
truncated. A 4000-`max_tokens` request came back `finish_reason: stop` with 1141
completion tokens — the model chose to stop. What actually failed was the *prompt*
side: an 11017-token request got

```
HTTP 400 {"message":"request (11017 tokens) exceeds max_total_tokens (4096)"}
```

So the ceiling was `--max-ctx 4096`, hard-coded at `scripts/serve_v100.sh:51` and
therefore **in git** — not, as first reported to the peer, "only a launch parameter,
no PR needed." I had read the running process's `/proc/<pid>/cmdline` and stopped
before reading where that value came from.

## What Worked

`--max-ctx 4096` → `32768`, an **8x** raise with no code change, no memory lever,
and nothing traded.

The reason it was free is the shape of `_fit_blocks` (`engine.py:1170`):

```python
return min(fit, cap) if cap else fit
```

`cap` is `--max-ctx × --max-batch / 16` (`cli.py:89`); `fit` is two thirds of the
free memory at build time. At `--max-ctx 4096 --max-batch 1` the cap is 256 blocks,
and the live server reported `blocks_total: 256` — **numerically identical to the cap,
so the number alone could not say which operand bound it.** That ambiguity is the
whole reason 4096 survived: it looks like a measured ceiling.

It was the cap. The fit at `--slots 8` with a draft was measured at **2649 blocks =
42384 tokens** (#65, `2026-09-04-slots-default-8-not-16.md`) in a *costlier*
configuration than serve runs — `--depth 3`, whose per-step verify states scale with
`slots × width`. 32768 tokens is 2048 blocks. The cap still binds at 32K, with the fit
above it.

| `--max-ctx` | blocks (= cap at `--max-batch 1`) | pool @ 2.125 MiB |
|---:|---:|---:|
| 4096 (was) | 256 | 544 MiB |
| **32768 (now)** | **2048** | **4352 MiB** |
| 42384 | 2649 | 5629 MiB — the measured fit, `--depth 3` |

Block size is 2.125 MiB, not the 2.00 MiB `serve-v100.md` carried: 2.000 MiB of trunk
K+V (16 full-attn layers × 4 KV heads × 16 tokens × head_dim 256 × f32 × 2 planes)
**plus the draft's mirrored 1/16** at `--depth 1`, which `_fit_blocks` charges for and
the doc's table did not.

**A derivation I published and then withdrew, in this entry because the retraction
should travel with the number.** I first put the ceiling at 2453 blocks / 39248
tokens, from `25492 / 32768 MiB` resident read off the live server, backing out
≈ 7820 MiB free at startup. That is wrong twice over. `nvidia-smi` later read
**22136 MiB**, not 25492 — and more importantly the read happens *after* the pool is
allocated, whereas `_fit_blocks` runs *after* `torch.cuda.empty_cache()`
(`engine.py:1211`), whose own comment records that call as the difference between a
1024-token and a 62832-token context. A post-hoc `nvidia-smi` cannot see the memory
state the fit saw. The measured 2649 supersedes the derived 2453; the doc now cites
the measurement.

## Rule

When a limit and a derived cap are numerically equal, the number cannot say which one
binds — read the `min()`, not the reported total. And never infer build-time free
memory from a later `nvidia-smi`: `empty_cache()` runs in between and is worth 6x here.

## Results

| date | commit | machine | target | model | max ctx | blocks | KV pool |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-05 | (this) | V100-SXM2-32GB | cuda sm70 | 27B NVFP4, draft d1 | 4096 → **32768** | 256 → **2048** | 544 → **4352 MiB** |

Unmeasured, and the reason this entry is not fully Shipped: peak memory at 32K, the
first long request's JIT cost, and a >11017-token request returning 200. All three
need a restart at this change, and the pod's checkout is on `docs-disk-full` at
`550740a` where `serve_v100.sh:51` still reads 4096.

**A claim withdrawn from this section.** It first said those three were unmeasured
because "the resident server is mid-session and was not interrupted." That was
assumed, not read. `/health` reports `finished: 0`, `tokens_generated: 0`,
`running: 0` and the log has zero request lines, against a process start of
09:00:44 — **31.5 minutes of uptime and not one request served.** Nothing was in
session. The three numbers are unmeasured because I did not measure them, and the
premise that excused it was the flattering one, which is why it went unchecked.

That JIT number is also the **`kv_tier` cold-start baseline**, and `kv_tier`'s
before-arm must be **32768**, not 4096 — 8x of the headroom is a flag, and folding it
into offload's credit would overstate offload.

Raw artifacts: `~/serve70c.log` on the pod (`=== boot N ===` lines carry the sha);
`/health` `blocks_total` before and after.
