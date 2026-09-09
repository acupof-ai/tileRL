"""Per-request cost of the API surface, over the scripted engine.

The engine is canned, so this measures the HTTP+render+parse path only -- which
is exactly what tranche (b) changed. A real number needs the V100 (pending-remote).

    TILERL_TARGET=cpu uv run python3 scripts/bench_api_routes.py

Emits one request_overhead_ms record per route to
docs/experience/bench/measurements.jsonl (schema: docs/bench-schema.md).
"""

from __future__ import annotations

import argparse
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path[:0] = ["src", "packages/tilerl-kernels/src", "tests"]

import benchrec  # noqa: E402
import uvicorn  # noqa: E402
from test_server import _ByteTokenizer, _ScriptedEngine  # noqa: E402

from tilerl.server import create_app  # noqa: E402

REPLY = "weighing it up\n</think>\n\nThe answer is 4."
TOOL = ("</think>\n\nI will run it.\n<tool_call>\n<function=Bash>\n"
        "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {
    "name": "Bash", "description": "run a command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
N = 60


class _Engine(_ScriptedEngine):
    def submit(self, input_ids, params=None) -> int:
        prompt = self._tok.decode(list(input_ids))
        self._replies = [TOOL if "run ls" in prompt and "<tool_response>" not in prompt
                         else REPLY]
        return super().submit(input_ids, params)


def _serve():
    tok = _ByteTokenizer()
    app = create_app(_Engine(tok, []), tok, model_name="tilerl")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(400):
        if srv.started:
            return srv, f"http://127.0.0.1:{port}"
        time.sleep(0.05)
    raise RuntimeError("uvicorn did not start")


def timed(label: str, fn, common: dict) -> None:
    fn()  # warm the connection and the route's first-call imports
    ms = []
    for _ in range(N):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1e3)
    ms.sort()
    print(f"{label:<34} median {statistics.median(ms):6.2f} ms   "
          f"p90 {ms[int(0.9 * len(ms))]:6.2f}   min {ms[0]:6.2f}")
    rec = {
        "metric": "request_overhead_ms", "value": round(statistics.median(ms), 2),
        "unit": "ms", "shape": {"route": label},
        "warm": {"state": "warm", "compiles": 0},
        "n": N, "spread": round((ms[-1] - ms[0]) / statistics.median(ms), 4),
        **common,
    }
    rec["floor"] = benchrec.measured_best_floor(rec, lower_is_better=True)
    print(f"  record {benchrec.append(rec)} appended", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    benchrec.add_record_args(ap, default_target="cpu", default_device="cpu")
    args = ap.parse_args()
    args.model_name = "scripted"
    common = benchrec.record_common(args, build="eager")

    import anthropic
    import openai

    srv, base = _serve()
    oa = openai.OpenAI(base_url=f"{base}/v1", api_key="x", max_retries=0)
    an = anthropic.Anthropic(base_url=base, api_key="x", max_retries=0)
    msg = [{"role": "user", "content": "hi"}]
    think = {"chat_template_kwargs": {"enable_thinking": True}}

    print(f"n={N} per row, canned engine (no weights): HTTP + render + parse only\n")
    timed("chat non-stream", lambda: oa.chat.completions.create(
        model="tilerl", messages=msg, extra_body=think), common)
    timed("chat non-stream + tools", lambda: oa.chat.completions.create(
        model="tilerl", messages=[{"role": "user", "content": "run ls"}],
        tools=TOOLS, extra_body=think), common)
    timed("chat stream (drain)", lambda: [
        c for c in oa.chat.completions.create(
            model="tilerl", messages=msg, stream=True, extra_body=think)], common)
    timed("messages non-stream", lambda: an.messages.create(
        model="tilerl", max_tokens=64, messages=msg), common)
    timed("messages non-stream + thinking", lambda: an.messages.create(
        model="tilerl", max_tokens=64, messages=msg,
        thinking={"type": "enabled", "budget_tokens": 32}), common)
    timed("messages stream (drain)", lambda: [
        e for e in an.messages.create(model="tilerl", max_tokens=64, messages=msg,
                                      stream=True)], common)
    timed("responses non-stream", lambda: oa.responses.create(
        model="tilerl", input="hi", extra_body=think), common)
    timed("responses stream (drain)", lambda: [
        e for e in oa.responses.create(model="tilerl", input="hi", stream=True,
                                       extra_body=think)], common)
    srv.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
