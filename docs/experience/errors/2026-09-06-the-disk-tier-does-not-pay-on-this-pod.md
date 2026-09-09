# The disk tier does not pay on this pod — bandwidth measurements, 2026-09-06

**Status:** closed. The DRAM snapshot tier shipped (`--dram-bytes`, default 0);
the disk tier's value is on NVMe hosts, untested. These measurements are the
reason the disk tier was not built for this pod.

## Context

The KV tier design (`design-kv-tier.md`, deleted 2026-09-09) assumed a 2 GB/s NVMe
device for the disk tier. The real device is 10.6x slower and is not flash. Both
numbers below are measured, not assumed.

## Measurements

| | figure | how |
|---|---:|---|
| `/data00` sequential read | **189 MB/s** | `dd` 1 GiB `iflag=direct` |
| `/data00` sequential write | **229 MB/s** | `dd` 1 GiB `oflag=direct` |
| pinned DRAM→HBM | **11.52 GiB/s** | `copy_` of a 149.6 MiB snapshot, 20 iters |
| pinned HBM→DRAM | **12.26 GiB/s** | same, reversed |
| unpinned DRAM→HBM | **6.12 GiB/s** | same buffer without `pin_memory` |
| PCIe link | **Gen3 x16** | `nvidia-smi --query-gpu=pcie.link.gen.current` |
| host RAM | **31 GiB total, 25 available** | `free -g` |
| both block devices | **`ROTA 1`** — no NVMe | `lsblk -d -o NAME,SIZE,ROTA` |

Pinning is worth 1.88x on H2D and 2.76x on D2H.

An 11019-token prompt re-prefills in **163 s** (14.68 ms/token) — that is what any
tier has to beat.

## Why the disk tier loses on this pod

Host RAM is **31 GiB against a 32 GiB card**. The DRAM tier is not a large backing
store behind a small cache; it is slightly smaller than HBM. At the context we
serve, the entire DRAM tier holds under 6 entries. A disk tier behind that adds
capacity at 45x the latency, on a spinning device that is also where the checkpoint,
logs and `~/pytmp` live — and `/` was 100% full at measurement time.

Even at 189 MB/s the disk beats re-prefilling by 13-20x arithmetically, and the
breakeven is 57 tokens. The reason not to build it is not arithmetic — it is that
the DRAM tier already covers the workload (concurrent sessions > HBM snapshot
budget), and the disk tier's capacity-per-latency is poor on this specific
hardware.

## Rule

A tier's value is hardware-specific. 189 MB/s is what a `ROTA 1` device gives, not
what the feature is worth — every number the disk tier reports from this pod is a
lower bound on what an NVMe host does. Measure the device before designing against
an assumed one.
