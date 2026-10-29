"""Download a pinned ModelScope snapshot and verify its bytes and tensor index."""

import argparse
import concurrent.futures
import hashlib
import json
import struct
import urllib.parse
import urllib.request
from pathlib import Path


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download(model, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    base = f"https://modelscope.cn/api/v1/models/{model}"
    listing = json.loads(fetch(base + "/repo/files?Revision=master&Recursive=true"))
    assert listing["Success"], listing
    files = {f["Path"]: f for f in listing["Data"]["Files"] if f["Type"] == "blob"}
    manifest = {"model": model, "listing_url": base + "/repo/files?Revision=master&Recursive=true",
                "files": files}
    (root / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def get(name):
        entry = files[name]
        assert Path(name).name == name, f"Unexpected nested path: {name}"
        dest = root / name
        if dest.exists():
            assert dest.stat().st_size == entry["Size"], f"Existing file has wrong size: {dest}"
            digest = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
            assert digest == entry["Sha256"], f"Existing file has wrong hash: {dest}"
            return
        url = base + "/repo?" + urllib.parse.urlencode({"Revision": entry["Revision"], "FilePath": name})
        part = dest.with_name(dest.name + ".partial")
        digest = hashlib.sha256()
        if entry["Size"] > 64 * 1024 * 1024:
            chunk = 64 * 1024 * 1024
            start_offset = part.stat().st_size if part.exists() else 0
            assert start_offset <= entry["Size"]
            def segment(start):
                end = min(start + chunk, entry["Size"]) - 1
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    assert response.status == 206
                    assert response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/")
                    payload = response.read(end - start + 2)
                assert len(payload) == end - start + 1
                with part.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(payload)
            if not part.exists():
                part.touch()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(segment, range(start_offset, entry["Size"], chunk)))
            with part.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256")
        else:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as target:
                while data := response.read(8 * 1024 * 1024):
                    target.write(data)
                    digest.update(data)
        assert part.stat().st_size == entry["Size"], (name, part.stat().st_size, entry["Size"])
        assert digest.hexdigest() == entry["Sha256"], f"Hash mismatch: {name}"
        part.rename(dest)
        print(json.dumps({"file": name, "bytes": entry["Size"], "sha256_ok": True}), flush=True)

    for name in ("config.json", "model.safetensors.index.json", "README.md"):
        get(name)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    expected = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                "vocab_size": 248320}
    assert all(text.get(k) == v for k, v in expected.items()), text
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted(set(index.values()))
    names = shards + [n for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                  "vocab.json", "merges.txt", "generation_config.json") if n in files]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(get, names))
    count = 0
    for name in shards:
        with (root / name).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            assert 0 < size < 128 * 1024 * 1024, (name, size)
            header = json.loads(stream.read(size))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        end = max(v["data_offsets"][1] for v in tensors.values())
        assert 8 + size + end == (root / name).stat().st_size, name
        for key, shard in index.items():
            if shard == name:
                assert key in tensors, (key, name)
                count += 1
    result = {"model": model, "verified": True, "indexed_tensors": count, "shards": shards}
    (root / "VERIFIED.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    download(args.model, args.output)
