"""Verify the persistent-grid primitive before compiling a full-model kernel."""

import json

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[1], target={"kind": "cuda", "arch": "sm_70"})
def grid_probe(blocks: int, rounds: int):
    @T.prim_func
    def main(A: T.Tensor((blocks, 128), "float32"), B: T.Tensor((blocks, 128), "float32")):
        with T.Kernel(blocks, threads=128) as bx:
            tx = T.get_thread_binding(0)
            B[bx, tx] = A[bx, tx]
            T.sync_grid()
            for step in T.serial(rounds):
                value = B[(bx + 1) % blocks, tx]
                T.sync_grid()
                B[bx, tx] = value + 1.0
                T.sync_grid()
    return main


def main():
    blocks = torch.cuda.get_device_properties(0).multi_processor_count
    rounds = 17
    x = torch.arange(blocks * 128, device="cuda", dtype=torch.float32).reshape(blocks, 128)
    kernel = grid_probe(blocks, rounds)
    expected = torch.roll(x, -rounds, 0) + rounds
    for _ in range(3):
        actual = kernel(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    failed = False
    try:
        torch.testing.assert_close(actual, expected + 1, rtol=0, atol=0)
    except AssertionError:
        failed = True
    assert failed, "The negative control did not fail"
    print(json.dumps({"gate": "persistent_grid", "blocks": blocks, "rounds": rounds,
                      "exact": True, "negative_control_rejected": failed}))


if __name__ == "__main__":
    main()
