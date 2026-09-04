"""Data parallelism: one engine per device, requests routed to the shortest queue.

The 27B in NVFP4 is ~23 GB against a 96 GB card, so replicating buys aggregate
throughput for memory the cards already have. Eight independent processes
measured 7.54x on 8 H20s (docs/experience/wins/2026-08-29-data-parallel-scales.md),
the ceiling an in-process wrapper approaches. Same submit/step/poll seam as
:class:`~tilerl.engine.Engine`.

# ponytail: one process, N CUDA contexts — the GIL serialises the Python half
# of each tick; the upgrade is a process per device with the router in front.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch


def _on(device) -> Any:
    if device is None or torch.device(device).type != "cuda":
        return contextlib.nullcontext()  # torch.cuda.device(None) raises on the CPU target
    return torch.cuda.device(device)


class DataParallelEngine:
    """N engines on N devices behind one submit/poll/step. Request ids are
    ``rank + n_ranks * inner_id`` so :meth:`poll` demuxes without a side table."""

    def __init__(self, engines: list[Any], devices: list[torch.device]):
        if not engines:
            raise ValueError("DataParallelEngine needs at least one engine")
        self._engines = engines
        self._devices = devices
        self._n = len(engines)

    @classmethod
    def build(cls, devices: list[int], make_engine, **kw) -> DataParallelEngine:
        # The Backend binds torch.cuda.current_device() on construction, so each
        # replica is built inside its own context or it shares device 0's pools.
        engines, devs = [], []
        for d in devices:
            with _on(torch.device("cuda", d)):
                engines.append(make_engine(d, **kw))
                devs.append(torch.device("cuda", d))
        return cls(engines, devs)

    def submit(self, input_ids: Any, params: Any = None) -> int:
        i = min(range(self._n), key=lambda j: self._engines[j].stats()["running"])
        with _on(self._devices[i]):
            rid = self._engines[i].submit(input_ids, params)
        return i + self._n * rid

    def step(self) -> None:
        for i, e in enumerate(self._engines):
            with _on(self._devices[i]):
                e.step()

    def poll(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, e in enumerate(self._engines):
            for rid, toks in e.poll().items():
                out[i + self._n * rid] = toks
        return out

    def logprobs(self, request_id: int) -> list[float] | None:
        i = request_id % self._n
        return self._engines[i].logprobs(request_id // self._n)

    def take(self, request_id: int) -> list[int] | None:
        i = request_id % self._n
        return self._engines[i].take(request_id // self._n)

    def peek(self, request_id: int) -> list[int] | None:
        i = request_id % self._n
        return self._engines[i].peek(request_id // self._n)

    def stats(self) -> dict[str, Any]:
        per = [e.stats() for e in self._engines]
        total = {k: sum(s[k] for s in per) for k in per[0] if isinstance(per[0][k], int)}
        total["replicas"] = self._n
        total["running_per_replica"] = [s["running"] for s in per]
        return total

    def is_idle(self) -> bool:
        return all(e.is_idle() for e in self._engines)

    def precapture(self) -> int:
        total = 0
        for i, e in enumerate(self._engines):
            with _on(self._devices[i]):
                total += e.precapture()
        return total

    def run(self) -> None:
        for i, e in enumerate(self._engines):
            with _on(self._devices[i]):
                e.run()

    def shutdown(self, timeout: float = 5.0) -> None:
        for e in self._engines:
            e.shutdown(timeout)


if __name__ == "__main__":  # runnable check: routing and id demux, no GPU needed
    class _Fake:
        def __init__(self, tag):
            self.tag, self.n, self.done = tag, 0, {}

        def stats(self):
            return {"running": self.n}

        def submit(self, ids, params=None):
            self.n += 1
            self.done[self.n - 1] = [self.tag, len(ids)]
            return self.n - 1

        def poll(self):
            d, self.done = self.done, {}
            return d

    dp = DataParallelEngine([_Fake(0), _Fake(1), _Fake(2)], [None] * 3)
    ids = [dp.submit([7] * (i + 1)) for i in range(6)]
    assert ids == [0, 1, 2, 3, 4, 5], ids  # round-robin while queues are equal
    got = dp.poll()
    for gid, (tag, _) in got.items():
        assert gid % 3 == tag, (gid, tag)
    assert len(got) == 6, got
    print("parallel: routing + id demux OK")
