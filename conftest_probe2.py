import collections, traceback, torch
from tilerl_kernels.backend import Backend
SITES = collections.Counter()
def _counting(t):
    if t.is_contiguous():
        if t.storage_offset() != 0:
            st = traceback.extract_stack()[-4:-1]
            SITES[" <- ".join("%s:%d %s" % (f.filename.split('/')[-1], f.lineno, f.name) for f in reversed(st))] += 1
            return t.clone()
        return t
    return t.contiguous()
Backend._c = staticmethod(_counting)
def report():
    print("\n=== offset-clone call sites ===")
    for k, n in SITES.most_common(5): print("x%-4d %s" % (n, k))
import atexit; atexit.register(report)
