"""Data parallelism: one engine per device, requests routed to the shortest queue.

The 27B in NVFP4 is ~23 GB against a 96 GB card, so the model fits on ONE
device. That is why data parallelism comes first here and tensor/pipeline
parallelism do not: sharding a model that fits buys latency and pays
communication, while replicating it buys aggregate throughput and pays only
memory the cards already have. Measured before this module existed — eight
independent processes reach 7.54x on 8 H20s (docs/experience/wins/
2026-08-29-data-parallel-scales.md), which is the ceiling any in-process
wrapper can approach but not beat.

The seam does not move: :class:`DataParallelEngine` offers the same
``submit`` / ``step`` / ``poll`` an :class:`~tilerl.engine.Engine` does, so a
server or an OPD loop cannot tell the difference.

# ponytail: one process, N CUDA contexts — the GIL serialises the Python half
# of each tick. A captured decode tick is one replay, so that half is small;
# if it stops being small, the upgrade path is a process per device with the
# router in front, which is what the 7.54x measurement already used.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch

__all__ = ["DataParallelEngine"]


def _on(device) -> Any:
    """Device context, or nothing on a non-CUDA target — the CPU path is the
    project's default and torch.cuda.device(None) raises there."""
    if device is None or torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(device)


class DataParallelEngine:
    """N engines on N devices behind one submit/poll/step.

    Request ids are ``rank + n_ranks * inner_id`` so :meth:`poll` can demux
    without a side table.
    """

    def __init__(self, engines: list[Any], devices: list[torch.device]):
        if not engines:
            raise ValueError("DataParallelEngine needs at least one engine")
        self._engines = engines
        self._devices = devices
        self._n = len(engines)

    @classmethod
    def build(cls, devices: list[int], make_engine, **kw) -> "DataParallelEngine":
        """``make_engine(device_index, **kw)`` runs under that device's context.

        The Backend binds ``torch.cuda.current_device()`` when it is
        constructed, so each replica must be built inside its own context —
        one built outside would silently share device 0's pools.
        """
        engines, devs = [], []
        for d in devices:
            with _on(torch.device("cuda", d)):
                engines.append(make_engine(d, **kw))
                devs.append(torch.device("cuda", d))
        return cls(engines, devs)

    def submit(self, input_ids: Any, params: Any = None) -> int:
        """Admit to the shortest queue; returns a global request id."""
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
        """Demux to the replica that served this id."""
        i = request_id % self._n
        return self._engines[i].logprobs(request_id // self._n)

    def take(self, request_id: int) -> list[int] | None:
        i = request_id % self._n
        return self._engines[i].take(request_id // self._n)

    def stats(self) -> dict[str, Any]:
        """Summed counters, plus the per-replica running counts."""
        per = [e.stats() for e in self._engines]
        total = {k: sum(s[k] for s in per) for k in per[0] if isinstance(per[0][k], int)}
        total["replicas"] = self._n
        total["running_per_replica"] = [s["running"] for s in per]
        return total

    def is_idle(self) -> bool:
        return all(e.is_idle() for e in self._engines)


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
    # every global id demuxes back to the replica that served it
    for gid, (tag, _) in got.items():
        assert gid % 3 == tag, (gid, tag)
    assert len(got) == 6, got
    print("parallel: routing + id demux OK")
