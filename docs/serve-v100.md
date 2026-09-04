# Serving the 27B from the V100, with the web UI on your Mac

The engine runs on the pod (the checkpoint and the GPU are there); the browser
runs here.

## 1. Start the server on the pod

```bash
ssh v100
cd ~/tilerl-git
setsid nohup scripts/serve_v100.sh >/dev/null 2>&1 &
```

`scripts/serve_v100.sh` carries the settled flags and restarts the server if it
dies — capped at 10, then it stays dead, because a server that restarts forever on
an OOM hides the OOM. `setsid` matters: without it the supervisor is killed when
the ssh session ends (`loginctl show-user` reports `Linger=no` on this pod, which
is also why this is a bash loop and not a systemd unit).

It logs to `~/serve70c.log`, one `=== boot N at <time> sha <rev> dirty <n> ===`
line per start, and the log is truncated past 32 MiB. A model load takes ~36 s;
`ss -ltn | grep :8000` tells you when it is up.

To stop it, TERM the supervisor and the server goes with it (verified: one TERM,
both processes gone in 2 s, card released):

```bash
ss -ltnp | grep :8000                    # the server's pid, from the socket that owns it
awk '{print $4}' /proc/<server-pid>/stat # its parent, the supervisor
kill -TERM <supervisor-pid>
```

Read the pid off the listening socket rather than matching a command line:
`pgrep -f serve_v100.sh` also matches the ssh command you typed to run it, so the
count it returns is not the number of supervisors.

## 2. Open it

The server binds `0.0.0.0`, so from this Mac the pod's address works directly:

```
http://10.37.2.27:8000/
```

`/` is the playground, `/about` is what tileRL is, `/health` has queue depth,
block usage, prefix-cache hits and the spec counters.

Two things that cost time to find out: the pod's FQDN has an **AAAA record that
does not answer** (8 s timeout), so use the bare IPv4; and a stale local server on
`127.0.0.1:8000` will answer instead of the tunnel if you have one, with a
different model and garbage output.

An SSH tunnel still works if you would rather not reach the pod directly:

```bash
ssh -N -L 8000:10.37.2.27:8000 v100
```

Note the pod-side address is **not** `127.0.0.1` — the server does not bind
loopback.

## Flags, and why

Set in `scripts/serve_v100.sh`, each from a measurement:

| flag | why |
|---|---|
| `--depth 1` | beats the shipped 3 by 1.204x at B=1 (#22/#72) |
| `--max-batch 1` | B=8 only survives while no prefill is in flight (#42) |
| `--max-ctx 4096` | the KV pool sizes itself from the config's 262144 otherwise |
| `--slots 8` (default) | 16 costs 5.19x the KV pool (#65) |

**`--max-ctx` is not optional on this card.** The 27B's config says 262144 tokens
of context and the pool sizes itself from that: f32 KV for the full window is
**275 GB**. The card has 32, of which 19 are already weights.

The server prints `N decode graphs in Ms` before the URL: every (batch bucket ×
chain width) a decode tick can key on is captured up front, so the first message
starts at the plateau instead of paying ~14 s per capture. `--no-warmup` opts out.
Note the graph covers the shapes capture *visited* — a request can still introduce
one it did not, which is what a first-visit compile spike is.

## Rates

Measured on the live server, 2026-09-04, from the pod so RTT is outside the window:

| window | tok/s |
|---|---:|
| decode-only (first token → last) | **50.0** |
| wall (request sent → last token) | 46.3 |

Both are right; they answer different questions, and the page's live counter shows
the first. **`wall_ms / tokens` is neither** — it charges prefill to decode, reads
~40, and has already been mistaken for a 15% regression that did not exist. Use
`scripts/probe_page_rate.py`, which prints both windows from one request.

Benched numbers for reference (direct `step()` loop, fully warm, `--depth 3`):

| ctx | dense tok/s | spec tok/s | prefill ms/token |
|---:|---:|---:|---:|
| 512 | 41.0 | 48.4 | 7.89 |
| 1024 | 40.4 | **50.8** | 8.03 |
| 2048 | 39.4 | 44.0 | 8.33 |
| 4096 | 37.6 | 40.3 | 8.92 |

## Environment, if you start it by hand

The supervisor sets these; you need them only outside it.

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH      # the default nvcc is 11.8 and rejects -std=c++20
export TILERL_TARGET=cuda
export TMPDIR=$HOME/pytmp TMP=$TMPDIR TEMP=$TMPDIR
export PYTHONPATH=$HOME/tilerl-git/src:$HOME/tilerl-git/packages/tilerl-kernels/src
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
~/venv70/bin/python -u -m tilerl.cli serve --model qwen38-27b \
    --draft $TILERL_QWEN38_SOURCE/model-00018-of-00018.safetensors --depth 1 \
    --host 0.0.0.0 --port 8000 --max-batch 1 --max-ctx 4096
```

`~/venv70/bin/python` has torch 2.5.1+cu121, which matches the 535 driver.

## How long a context fits, and what buys more

The limit is KV bytes, not the model. One 16-token block holds f32 K and V for 16
full-attention layers × 4 KV heads × head_dim 256 = **2.00 MiB**, and the pool has
to cover `max_batch` rows of the context you admit.

Headroom after weights, states and allocator slack is about **7.5 GiB**:

| ctx | blocks @ max_batch 2 | f32 pool | fits |
|---:|---:|---:|:--|
| 4096 | 512 | 1.00 GiB | yes |
| 8192 | 1024 | 2.00 GiB | yes |
| 16384 | 2048 | 4.00 GiB | yes |
| 32768 | 4096 | 8.00 GiB | no, just over |
| 65536 | 8192 | 16.00 GiB | no |

So **16K works today** with `--max-ctx 16384`, and 32K needs one of the levers
below. `--blocks` no longer has to be passed: the pool is sized from `--max-ctx`
and `--max-batch` together — the live server logs `blocks = 4096 tokens` for
`--max-ctx 4096 --max-batch 1`, which is 256 blocks, half the table's row because
the table assumes `--max-batch 8`. That multiplier is the point: dropping 8 → 2 is
what moved the ceiling from 4K to 16K, and `--max-batch 1` doubles it again.

Three levers, cheapest first:

1. **`--max-batch 1`** — halves the pool, so 32K fits. Free for one person; costs
   concurrency you were not using.
2. **An f16 KV pool** — halves the block to 1.00 MiB, so 32K costs 4 GiB and 64K
   costs 8. This is the real fix and it is *not* free: the pool dtype IS the
   attention kernel's ABI (`wins/2026-09-02-kv-pool-dtype-is-the-kernel-abi.md` —
   a bf16 pool against the f32 kernel cast the whole plane every call, 4.71
   ms/token). It needs an f16 `paged_attention_split`, its own parity run, and a
   check that f16 K/V does not degrade long-range attention. Not done.
3. **`--kv-tier <dir>`** — spill cold blocks to disk (`KvTier`, wired to `serve`
   here). Trades prefix hit rate for capacity, so it suits long documents that
   get re-read rather than one long generation. The engine path is tested; the
   capacity it buys on this card is unmeasured.

The table above is arithmetic from the block size, not a measured sweep — 4K is
the only row actually served end to end so far. Treat the rest as what to try,
and expect the allocator to want more slack than the ideal number.

Not a lever: the GDN layers. 48 of the 64 layers are gated-delta and carry a
fixed-size recurrent state, so their cost does not grow with context at all —
which is why this model's context is cheaper than a 64-layer full-attention model
of the same size.

## Startup is slow, and that is expected

Weight load is ~5 minutes on a cold page cache (19 GB from disk, dequantized and
twiddled); warm it is ~36 s to listening. Then the warmup captures graphs: cold JIT
is 30-120 s per kernel shape, and the tilelang cache (`~/.tilelang/cache`) makes the next run
~0.2 s each. All of it happens before the URL is printed.

A 4096-token prompt then takes ~36 s to first token (7.89-8.92 ms per prompt
token) — the same figure as a warm load, coincidentally, and a different thing.
Generation rate: the Rates section above, not the bench table.

If startup looks hung, it is compiling. `tail -f` the server's output to watch.

## What the numbers should look like

Four equal-length requests through HTTP on a cold server, decode dominated:

| warmup | cost | req 1 | req 2 | req 3 | req 4 |
|---|---:|---:|---:|---:|---:|
| `--no-warmup` | — | 0.9 | 2.6 | 2.9 | 8.3 |
| **default (precapture)** | **19 s** | **13.6** | 21.2 | 24.3 | 25.5 |
| `bench_ctx_decode`, 512 ctx d3 | | | | | 48.4 |

Nothing degrades and nothing leaks — free HBM was flat at 2.81 GiB across a
six-request run. Two separate costs are being paid down here:

1. **CUDA graph capture**, ~14 s each, one graph per (batch bucket × chain
   width). `precapture` does all 8 up front, which is why request 1 is 15×
   better than with no warmup.
2. **fp4 kernel JIT**, which specializes per prefill shape. That is the residual
   13.6 → 25.5 ramp, with every graph already resident. It settles after a few
   messages, and the tilelang cache (`~/.tilelang/cache`, the default — nothing
   sets `TILELANG_CACHE_DIR`) keeps it across restarts.

## If it OOMs anyway

Lower `--max-ctx` first — it is what sizes the pool now, and it also caps what a
request may ask for, so nothing can outgrow the pool you sized. Then `--max-batch`,
which multiplies it. `--slots` last: the live server runs the default 8 and each GDN
state slot also owns the per-step verify states when a draft is loaded, so 16 costs
5.19x the pool (#65) and there is little to reclaim below 8.

`--blocks` still exists and still overrides the derived count, but the live config
does not pass it; prefer the two knobs above so the number stays derived from what
you actually want to serve.

## One job at a time

`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before starting.
A leftover process holds its memory invisibly to `--query-gpu`, and killing by
pattern has orphaned GPU memory here before — find the pid, verify it with
`ls -l /proc/<pid>/fd | grep nvidia`, then kill that pid.
