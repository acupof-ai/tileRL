"""Why does the SSE stream still coalesce? Count peek() hits under the real driver."""
import json

from fastapi.testclient import TestClient

from tests.test_server import _build_engine, _ByteTokenizer, create_app


def main() -> None:
    engine = _build_engine(seed=42)
    calls = {"peek": 0, "none": 0, "lens": []}
    real_peek = engine.peek

    def counted(rid):
        calls["peek"] += 1
        out = real_peek(rid)
        if out is None:
            calls["none"] += 1
        else:
            calls["lens"].append(len(out))
        return out

    engine.peek = counted
    engine.run()
    app = create_app(engine, _ByteTokenizer())
    try:
        with TestClient(app) as c:
            r = c.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 24,
                    "temperature": 0.0,
                    "seed": 7,
                    "stream": True,
                },
            )
        lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
        payloads = [json.loads(ln[6:]) for ln in lines[:-1]]
        deltas = [
            p["choices"][0]["delta"]["content"]
            for p in payloads
            if p["choices"][0].get("delta", {}).get("content")
        ]
        print(f"deltas={len(deltas)} peek_calls={calls['peek']} none={calls['none']}")
        print(f"peek lens: {calls['lens']}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
