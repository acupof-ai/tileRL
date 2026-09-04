"""How many bytes does silu_mul copy, and would a [2, M, I] layout remove them?

The fused gate_up writes [M, 2I], so `gate` and `up` are strided slices and
backend.silu_mul's `_c()` materialises both. Measures the copy on the real 27B MLP shape
rather than trusting the 476 MiB in the notes.
"""

import os

os.environ.setdefault("TILERL_TARGET", "cpu")

import torch

from tilerl.config import qwen38_27b


def main() -> None:
    cfg = qwen38_27b()
    inter = cfg.intermediate_size
    print(f"intermediate_size={inter} hidden={cfg.hidden_size} layers={cfg.num_layers}")

    for rows in (1, 2, 4, 8, 32, 512, 3584):  # 3584 = 7 x 512, the shape that OOMed
        gu = torch.empty(rows, 2 * inter, dtype=torch.float32)
        gate = gu[..., :inter]
        up = gu[..., inter:]
        per_half = rows * inter * 4
        copied = (0 if gate.is_contiguous() else per_half) + (
            0 if up.is_contiguous() else per_half
        )
        print(
            f"rows={rows:5d}  gate_contig={gate.is_contiguous()!s:5s} "
            f"up_contig={up.is_contiguous()!s:5s}  gu={gu.numel() * 4 / 2**20:8.1f} MiB  "
            f"copied={copied / 2**20:8.1f} MiB"
        )
    # Layout [2, M, I]: each half is its own contiguous plane at every row count.
    for rows in (1, 8, 3584):
        alt = torch.empty(2, rows, inter, dtype=torch.float32)
        assert alt[0].is_contiguous() and alt[1].is_contiguous(), rows
    print("[2, M, I]: both halves contiguous at every row count checked")


if __name__ == "__main__":
    main()
