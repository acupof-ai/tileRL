import collections, torch
from tilerl_kernels.backend import Backend
STATS = collections.Counter(); SHAPES = collections.Counter()
def _counting(t):
    if t.is_contiguous():
        STATS["contig"] += 1
        if t.storage_offset() != 0:
            STATS["contig_offset"] += 1
            SHAPES[(tuple(t.shape), str(t.dtype))] += 1
            return t.clone()
        return t
    STATS["noncontig"] += 1
    return t.contiguous()
Backend._c = staticmethod(_counting)
def report():
    print("\nCONTIG_OFFSET_CLONES=%d  contig0=%d noncontig=%d" % (
        STATS["contig_offset"], STATS["contig"] - STATS["contig_offset"], STATS["noncontig"]))
    for k, n in SHAPES.most_common(8):
        print("  x%-4d %s %s" % (n, k[0], k[1]))
import atexit; atexit.register(report)
