"""Which maker did each kernel dispatch actually resolve to? Runs another script
unchanged in this process, counting every ``Backend._kernel`` call by the maker
the registry resolved -- a name is shared between cells, a maker is not.

py-spy cannot answer this in scripts/prof_prefill_ops.py: that script syncs after
every op, so the CPU stack is inside torch.cuda.synchronize for nearly the whole
wall clock and the dispatch branch is never on it. Blind to a CUDA graph replay,
which runs no Python -- prefill is never captured, decode is.

    # negative control on this machine: arch=cpu, so the sm70 branch cannot run
    TILERL_TARGET=cpu uv run python scripts/prove_kernel_routing.py \\
        scripts/prof_prefill_ops.py --selfcheck
"""

from __future__ import annotations

import runpy
import sys
from collections import Counter

sys.path.insert(0, "src")
sys.path.insert(0, "packages/tilerl-kernels/src")

from tilerl_kernels.backend import Backend  # noqa: E402
from tilerl_kernels.registry import _resolve  # noqa: E402

_EXPECT = "kernels_attn.make_paged_attention_prefill_sm70"


def _maker(b, name: str) -> str:
    try:
        fn = _resolve(b.precision, b.arch)[name]
    except Exception:
        return "<unresolved>"
    mod = getattr(fn, "__module__", "?").rsplit(".", 1)[-1]
    return f"{mod}.{getattr(fn, '__name__', '<lambda>')}"


def _report(seen: Counter, expect: str) -> int:
    print("\n== dispatches through Backend._kernel (count, arch, op, maker, factory args) ==")
    for (arch, op, maker, variant), n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"{n:>7}  {arch:<5} {op:<30} {maker:<48} {variant}")
    if not seen:
        print("INSTRUMENT SAW NOTHING: Backend._kernel was never called, so this run "
              "says nothing about routing -- the patch or the script did not reach a kernel")
        return 0
    hits = sum(n for (_, _, maker, _), n in seen.items() if expect in maker)
    print(f"\nexpect maker {expect}: {hits} of {sum(seen.values())} dispatches")
    print(f"PROVEN: {expect} carried {hits} dispatch(es)" if hits else
          f"the sm70 cell did not run: no dispatch resolved to {expect}")
    return hits


def main() -> int:
    argv, expect = sys.argv[1:], _EXPECT
    if argv[:1] == ["--expect"]:
        expect, argv = argv[1], argv[2:]
    if not argv:
        print("usage: prove_kernel_routing.py [--expect MAKER] <script.py> [args...]")
        return 2
    seen: Counter = Counter()
    inner = Backend._kernel

    # counted per dispatch, not per compile: _kernel is called on every call and
    # caches the compiled kernel behind the same key
    def counting(self, name, *a, **kw):
        seen[(self.arch, name, _maker(self, name), f"{a}{sorted(kw.items())}")] += 1
        return inner(self, name, *a, **kw)

    Backend._kernel = counting
    sys.argv, rc = argv, 0
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else int(e.code is not None)
    finally:
        hits = _report(seen, expect)
    return rc or (0 if hits > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
