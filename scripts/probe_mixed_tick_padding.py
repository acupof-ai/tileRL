"""How much of the sm70 attention partials buffer is mixed-tick padding?

engine.py:724-729 gives every row in a tick ONE bucketed width, so a tick carrying
decode rows plus a long prefill chunk inflates the decode rows to the chunk's width.
paged_attention_split then allocates PO [B, S, H, KVSPLIT, D] f32 over that padded
shape (kernels.py:657) -- 4 rows x S=512 x 24 x 32 x 256 x 4 B = exactly the 1.500 GiB
that OOMs B=8 at ctx=512.

Runs on the CPU target: this is scheduler bookkeeping, not a kernel measurement.
"""

from __future__ import annotations

from tilerl_kernels.backend import get_backend

from tilerl.cli import _build_model
from tilerl.engine import _PREFILL_BUCKET, SamplingParams, build_engine

H, D, KVSPLIT = 24, 256, 32  # checkpoint config.json text_config + registry.py:127


def po_bytes(rows: int, width: int) -> int:
    return rows * width * H * KVSPLIT * D * 4


def main() -> None:
    cfg, model = _build_model("tiny", seed=0)
    e = build_engine(cfg, model, get_backend(), num_blocks=256, num_slots=12,
                     max_batch=8, max_total_tokens=512, max_num_batched_tokens=64)

    ticks: list[tuple[str, int, int, int]] = []  # kind, rows, width, useful positions
    orig = type(e)._run_forward

    def spy(self, decodes, prefills, chunks):
        rows = len(decodes) + len(prefills)
        seq_q = [1 + len(r.drafts) for r in decodes] + list(chunks)
        chunk = max(chunks, default=0)
        width = (-(-max(seq_q) // _PREFILL_BUCKET) * _PREFILL_BUCKET if chunk > 1
                 else max(seq_q))
        kind = "mixed" if decodes and prefills else ("decode" if decodes else "prefill")
        ticks.append((kind, rows, width, sum(seq_q)))
        return orig(self, decodes, prefills, chunks)

    type(e)._run_forward = spy
    rids = []
    for i in range(6):  # staggered arrivals, mixed prompt lengths
        rids.append(e.submit(list(range(3, 3 + 20 + i * 15)),
                             SamplingParams(temperature=0.0, max_new_tokens=8, seed=i)))
        for _ in range(5):
            e.step()
            e.poll()
    done: dict = {}
    for _ in range(600):
        e.step()
        done.update(e.poll())
        if all(r in done for r in rids):
            break
    type(e)._run_forward = orig

    by = {"decode": [], "prefill": [], "mixed": []}
    for kind, rows, width, useful in ticks:
        by[kind].append((rows, width, useful))
    print(f"{'kind':8} {'ticks':>6} {'mean rows':>10} {'mean width':>11} {'useful':>8} {'waste':>7}")
    for kind, xs in by.items():
        if not xs:
            print(f"{kind:8} {0:>6}")
            continue
        tot = sum(r * w for r, w, _ in xs)
        use = sum(u for _, _, u in xs)
        print(f"{kind:8} {len(xs):>6} {sum(r for r, _, _ in xs)/len(xs):>10.1f} "
              f"{sum(w for _, w, _ in xs)/len(xs):>11.1f} {use/tot:>7.0%} {1-use/tot:>7.0%}")

    mixed = by["mixed"]
    if mixed:
        worst = max(mixed, key=lambda x: x[0] * x[1])
        rows, width, useful = worst
        print(f"\nwidest mixed tick: {rows} rows x {width} = {rows*width} positions, "
              f"{useful} useful ({useful/(rows*width):.0%})")
        print(f"at the 27B's H={H} D={D} KVSPLIT={KVSPLIT}, that shape would allocate "
              f"{po_bytes(rows, width)/1024**3:.3f} GiB of PO")
        print(f"  4 rows x 512 (the shape that OOMs at ctx=512): "
              f"{po_bytes(4, 512)/1024**3:.3f} GiB, "
              f"{(1 - 515/(4*512))*100:.0f}% padding with 3 decodes + one 512 chunk")

    # The guard: a mixed tick must never be wider than its widest real row needs.
    assert by["mixed"], "no mixed tick occurred; this probe measured nothing"
    for rows, width, useful in mixed:
        assert useful < rows * width, f"a mixed tick with no padding: {rows}x{width}"


if __name__ == "__main__":
    main()
