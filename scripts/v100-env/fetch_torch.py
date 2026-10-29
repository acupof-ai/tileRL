"""Fetch a locked torch wheel through parallel byte ranges, without changing its hash."""

import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path


root = Path(__file__).resolve().parent
url = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl"
size = 780460642
output = root / "torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl"
part = output.with_suffix(".partial")
chunk = 16 * 1024 * 1024


def get(start):
    end = min(start + chunk, size) - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as stream:
        assert stream.status == 206
        assert stream.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
        data = stream.read(end - start + 2)
    assert len(data) == end - start + 1
    with part.open("r+b") as stream:
        stream.seek(start)
        stream.write(data)


part.touch(exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(get, range(0, size, chunk)))
with part.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == "c8ab8c92eab928a93c483f83ca8c63f13dafc10fc93ad90ed2dcb7c82ea50410"
assert part.stat().st_size == size
part.rename(output)
print(json.dumps({"file": str(output), "sha256": digest}), flush=True)
