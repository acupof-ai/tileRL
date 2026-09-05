"""Which of two contradicting numbers is wrong: 198.9 MiB/s or the bench's 1.732 s?

The restart bench's faulted arm reads a 309.6 MiB entry inside a 1.732 s wall that also
contains 169 tokens of prefill and 8 decode steps. A serial read of the whole spill
directory measured 198.9 MiB/s, which would put that one read at 1556 ms and leave 176 ms
for everything else -- against 185 ms for the tail prefill alone at the cold arm's rate.
The delta between a warm-cache and evicted-cache faulted arm (1.168 -> 1.732 s) implies
548.9 MiB/s instead, 2.8x apart.

Three candidates, measured here rather than argued:

  serial_all     the original probe, 14 files back to back -- reproduces or does not
  one_entry      just the two files a single fault-in reads, evicted first
  large_blocks   the same two files at 16 MiB reads instead of 4 MiB

If one_entry is much faster than serial_all, the 198.9 was the small files dragging the
average down and the bench's number stands. If one_entry reproduces 198.9, then the
bench's 1.732 s cannot contain a full-size read and something else is going on -- a
partial evict, or a read that overlaps GPU work.
"""

from __future__ import annotations

import os
import sys
import time

DIR = sys.argv[1] if len(sys.argv) > 1 else "/work/ssd_tier_bench/tilerl_kvtier"


def _evict(paths) -> None:
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _read(paths, block: int) -> tuple[float, int]:
    total = 0
    t0 = time.monotonic()
    for p in paths:
        with open(p, "rb") as fh:
            while chunk := fh.read(block):
                total += len(chunk)
    return time.monotonic() - t0, total


def _report(name: str, paths, block: int) -> None:
    _evict(paths)
    el, n = _read(paths, block)
    mib = n / 2**20
    print(f"{name:14} files={len(paths):2} {mib:8.1f} MiB in {el:6.3f} s = "
          f"{mib / el:7.1f} MiB/s (block {block >> 20} MiB)", flush=True)


def main() -> None:
    names = sorted(f for f in os.listdir(DIR) if f.endswith((".kv", ".st")))
    allp = [os.path.join(DIR, f) for f in names]
    # The servable entry is the second-largest .kv plus its .st sibling: the largest is
    # the whole prompt, which _match_prefix treats as a miss.
    kvs = sorted((os.path.getsize(os.path.join(DIR, f)), f) for f in names if f.endswith(".kv"))
    stem = kvs[-2][1][:-3]
    one = [os.path.join(DIR, stem + ext) for ext in (".kv", ".st")]

    _report("serial_all", allp, 1 << 22)
    _report("one_entry", one, 1 << 22)
    _report("large_blocks", one, 1 << 24)
    # And the same entry with the cache WARM, which is what the first bench measured.
    _read(one, 1 << 22)
    el, n = _read(one, 1 << 22)
    print(f"{'one_entry_warm':14} files={len(one):2} {n / 2**20:8.1f} MiB in {el:6.3f} s = "
          f"{n / 2**20 / el:7.1f} MiB/s (page cache holds it)", flush=True)


if __name__ == "__main__":
    main()
