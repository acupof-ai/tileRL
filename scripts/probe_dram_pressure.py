"""Which pressure evicted 27 entries per arm -- blocks, or state bytes?

`demote=0` with `evictions=27` in every arm says the tier was never called. There are two
eviction paths and only one of them can demote:

* `insert`'s tail loop, on `len > capacity or state_used > state_bytes` -- this one tries
  `_demote_one` first when a tier exists.
* `evict_until_free`, called when the BLOCK POOL cannot satisfy an allocation -- it calls
  `_evict_one` directly and cannot demote, by design: a snapshot tier cannot return blocks.

Reading the code says which is which; it does not say which one ran. So serve the same
prompts with the tier on and read `/health` after every turn, printing both operands of
each condition: `blocks_used`/`blocks_total` and `prefix_state_bytes` against the store's
byte budget. Whichever is at its ceiling when evictions jump is the pressure.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "scripts")
from bench_chat_interleaved import _fillers  # noqa: E402

PORT = 8124
LOG = "/work/dram_pressure.log"


def _stats():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10.0) as r:
        return json.loads(r.read())["stats"]


def main() -> int:
    sessions, turns, grow = 2, 3, 40
    cmd = [sys.executable, "-u", "-m", "tilerl.cli", "serve", "--model", "qwen38-27b",
           "--host", "127.0.0.1", "--port", str(PORT), "--max-batch", "1",
           "--max-ctx", "8192", "--slots", "3", "--dram-bytes", str(4 << 30)]
    env = dict(os.environ, TILELANG_CACHE_DIR="/work/tilelang_cache")
    with open(LOG, "wb") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    try:
        end = time.monotonic() + 900
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                st = _stats()
                break
            except (urllib.error.URLError, OSError, KeyError):
                time.sleep(1.0)
        else:
            raise TimeoutError("server not up")

        assert st.get("dram_budget") == (4 << 30), f"tier off: dram_budget={st.get('dram_budget')}"
        # `/health` does not publish the store's byte budget, so state-byte pressure is
        # read as prefix_state_bytes RISING and then flattening, not against a ceiling.
        print(f"pool: blocks_total={st['blocks_total']}", flush=True)
        # EVERY key, not a chosen subset. Three of the four rounds this probe took were
        # spent on a question `pool_used_blocks` already answered: it was in the response
        # the whole time and absent from the tuple I had listed, so the probe reported a
        # partition of the data rather than the data.
        keys = sorted(st)
        fillers = _fillers(sessions)
        convs = [[] for _ in fillers]
        for turn in range(turns):
            for c, filler in enumerate(fillers):
                convs[c].append({"role": "user", "content": filler * grow * (turn + 1)})
                before = _stats()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=json.dumps({"model": "qwen38-27b", "messages": convs[c],
                                     "max_tokens": 32, "temperature": 0.0}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    out = json.loads(r.read())
                after = _stats()
                convs[c].append({"role": "assistant",
                                 "content": out["choices"][0]["message"]["content"]})
                ev = after["prefix_evictions"] - before["prefix_evictions"]
                dm = after["dram_demotions"] - before["dram_demotions"]
                # Peak, not delta: the question is whether an operand REACHED its ceiling.
                print(f"turn {turn} c{c} tokens={out['usage']['prompt_tokens']:5d} "
                      f"evict+{ev:3d} demote+{dm:3d} | "
                      + "  ".join(f"{k}={after.get(k)}" for k in keys), flush=True)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
