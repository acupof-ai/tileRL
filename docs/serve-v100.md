# Serving the 27B from the V100, with the web UI on your Mac

The engine runs on the pod (the checkpoint and the GPU are there); the browser
runs here. One SSH tunnel connects them, so nothing is exposed off the pod.

## 1. Tunnel

```bash
ssh -N -L 8000:127.0.0.1:8000 v100
```

Leave it open. `-N` means no shell, just the forward. `127.0.0.1` on the pod side
matters: the server binds loopback only, so the tunnel is the only way in.

## 2. Start the server on the pod

```bash
ssh v100
cd ~/tilerl-v100
export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$HOME/tilerl-v100/src:$HOME/tilerl-v100/packages/tilerl-kernels/src
export TILELANG_CACHE_DIR=$HOME/.tilelang_cache
export TILERL_TARGET=cuda
CKPT=$HOME/models/Qwen3.8-27B-NVFP4
export TILERL_QWEN38_SOURCE=$CKPT

/usr/bin/python3 -c 'from tilerl.cli import main; main()' \
    serve --model qwen38-27b --port 8000 \
    --draft $CKPT/model-00018-of-00018.safetensors --depth 3 \
    --slots 4 --blocks 2048 --max-ctx 4096
```

It prints `N decode graphs in Ms` before the URL: every (batch bucket × chain
width) a decode tick can key on is captured up front, so the first real message
starts at the plateau instead of paying ~14 s per capture. Without it, serving
reads 1088 ms/token cold and reaches 26 only after about six requests.
`--no-warmup` opts out. `--max-batch` defaults to 2 — right for one person, and
it is also the size of the grid to capture and hold.

For reference, the benched numbers (direct `step()` loop, fully warm):

| ctx | dense tok/s | spec tok/s | prefill ms/token |
|---:|---:|---:|---:|
| 512 | 41.0 | 48.4 | 7.89 |
| 1024 | 40.4 | **50.8** | 8.03 |
| 2048 | 39.4 | 44.0 | 8.33 |
| 4096 | 37.6 | 40.3 | 8.92 |

Speculation is on by default (`--depth 3`) and wins from 512 tokens up; at very
short contexts it loses, because the draft costs more than a short trunk forward
saves. Drop `--draft` to compare.

`/usr/bin/python3`, not the venv: it has torch 2.5.1+cu121, which matches the 535
driver. nvcc must be 12.4 — the one on the default PATH is 11.8 and rejects
`-std=c++20`.

`-c 'from tilerl.cli import main; main()'` rather than `-m tilerl` or the `tilerl`
console script: the package ships no `__main__.py`, and the entry point pyproject
declares is installed into the venv, not into `/usr/bin/python3`.

**`--blocks 2048 --max-ctx 4096` is not optional on this card.** The 27B's config
says 262144 tokens of context, and the KV pool sizes itself from that: 131072
blocks of f32 KV is **275 GB**. The card has 32, of which 19 are already weights.
Serving is the only path that ever asked for the default — every bench script
passed `num_blocks` explicitly — so this was unexercised until now. 2048 blocks =
32768 tokens of KV = 4.3 GB, which serves 8 concurrent rows at 4K.

For 8K context: `--blocks 4096 --max-ctx 8192` (8.6 GB). Above that you are
trading against the weights.

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

So **16K works today** with `--max-ctx 16384 --blocks 2048`, and 32K needs one of
the levers below. Note `--max-batch` multiplies all of it: dropping 8 → 2 is what
moved the ceiling from 4K to 16K, and `--max-batch 1` doubles it again.

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

## 3. Open it

- <http://127.0.0.1:8000/chat> — the playground
- <http://127.0.0.1:8000/> — landing page
- <http://127.0.0.1:8000/health> — queue depth, block usage, prefix-cache hits,
  speculation accept rate

## Startup is slow, and that is expected

Weight load is ~5 minutes (19 GB from disk, dequantized and twiddled). Then the
warmup request compiles kernels and captures graphs: cold JIT is 30-120 s per
kernel shape, and `TILELANG_CACHE_DIR` on persistent storage makes the next run
~0.2 s each. All of that happens before the URL is printed.

A 4096-token prompt then takes ~36 s to first token (7.89-8.92 ms per prompt
token). Generation rate: see the table below, not the bench numbers.

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
   messages and `TILELANG_CACHE_DIR` keeps it across restarts.

## If it OOMs anyway

Lower `--blocks` first, then `--slots` (each GDN state slot also owns the
per-step verify states when a draft is loaded, so 4 is right for 32 GB, not the
16 a 96 GB card affords). `--max-ctx` caps what a request may ask for, so it
cannot outgrow the pool you sized.

## One job at a time

`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before starting.
A leftover process holds its memory invisibly to `--query-gpu`, and killing by
pattern has orphaned GPU memory here before — find the pid, verify it with
`ls -l /proc/<pid>/fd | grep nvidia`, then kill that pid.
