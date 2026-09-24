#!/bin/bash
# #805 production cutover window, 2026-09-24.
#
# $M = 82e3dfe3 (#818 merge). Engine runs from THIS tree's src/packages (HEAD
# is $M, so the engine and the tree agree by construction); the probe DRIVER is
# taken from perf1's sha 0bcc61fc, whose `git diff $M HEAD -- src packages` is
# empty. The driver is copied into the tree's scripts/ (never src/) and its
# content hash is checked against git's blob so "same script" is verified, not
# assumed.
#
# Order: kill the old serve -> ⑤④ correctness (graph vs forced-eager ref, full
# sequence) -> prefill tick-wall comparison -> start the $M service -> ①②③.
# ckl 2026-09-24: V100 is a test machine. No .bak, no old tree, no restore after
# every window; the end state is the latest main as the service. Fallback for
# this config is the SAME tree with `--sparse-min-tokens 8192` re-added, which
# sets sparse_graph_on=False (engine.py:763 `not self._sparse_min_tokens`) and
# reproduces today's behaviour.
set -x
TS=$(date +%m%d-%H%M%S)
D=$HOME/cutoverwin-$TS
mkdir -p "$D"
STAGE=$HOME/cutover-stage
# Stage files live in $STAGE; copy in the ones this run needs. The driver is
# copied to $D/probe_serve_sm70_w2048.py because the block below reads it there.
cp "$STAGE/probe_stage.py" "$D/probe_serve_sm70_w2048.py" || { echo "FATAL stage"; exit 90; }
cp "$STAGE/make_cutover_prompts.py" "$STAGE/parse_prefill_ticks.py" "$D/" || { echo "FATAL stage"; exit 90; }

T=$HOME/tilerl-v100-prod-82e3dfe3
M=82e3dfe359fe36cabdd63a3719daa479f4999258
DRIVER_SHA=0bcc61fc
OLD_PID=3587175
PROBE=$T/scripts/probe_serve_sm70_w2048.py

cd "$T" || exit 90
echo "=== TREE HEAD=$(git rev-parse HEAD)"
echo "=== SRC=$(git rev-parse HEAD:src)"
[ "$(git rev-parse HEAD)" = "$M" ] || { echo "FATAL tree != \$M"; exit 90; }

# The driver is staged by the caller into $D before this script runs.
cp "$D/probe_serve_sm70_w2048.py" "$PROBE" || { echo "FATAL driver missing"; exit 90; }
# Pin by CONTENT, not by branch: the $M worktree was never fetched the probe
# branch, so `git rev-parse 0bcc61fc:...` fails there. This literal is
# `git hash-object` of `0bcc61fc:scripts/probe_serve_sm70_w2048.py`, read from
# the repo that has the branch. Both sides must be blob hashes: comparing a blob
# hash to a raw `sha1sum` is a mismatch that looks like a bad copy.
DRIVER_BLOB=189e5b718861aeaf3b1465033d4432ea405f4e0c
GOT=$(git hash-object "$PROBE")
echo "=== DRIVER blob=$GOT want=$DRIVER_BLOB"
[ "$GOT" = "$DRIVER_BLOB" ] || { echo "FATAL driver blob mismatch"; exit 90; }
# src/packages identity: also by content. 0bcc61fc is not in this worktree, so
# compare against $M itself — the claim is "the engine here is $M's engine", and
# perf1's empty-diff result (0bcc61fc vs $M over src/packages) is what carries it
# across; re-verify the empty diff on the branch's own tree, not here.
echo "=== HEAD is $M: $(git rev-parse HEAD)"
echo "=== worktree src/packages dirty? (must be empty):"
git status --porcelain -- src packages
[ -z "$(git status --porcelain -- src packages)" ] || { echo "FATAL src dirty"; exit 90; }
# The probe's own tree assert must name THIS tree ($M), not the driver's branch:
# the engine is $M's and the driver's identity is pinned above by content hash.
# Passing another sha here would fail the probe's `rev-parse HEAD` check.
EXPECT_TREE=$M

export PATH=/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$T/src:$T/packages/tilerl-kernels/src
export TILERL_TARGET=cuda TILELANG_CACHE_DIR=$HOME/.tilelang_cache TMPDIR=$HOME/tmp
export TILERL_QWEN38_SOURCE=$HOME/models/Qwen3.8-27B-NVFP4
export H2_COLD_BYTES=1073741824 H2_COLD_SSD=$HOME/sparse_cold_128k.bin
export H2_COLD_SSD_BYTES=8589934592 H2_COLD_FORMAT=f16
export TILERL_STEP_TIMING=1 TILERL_STEP_TIMING_SLOW_MS=0
mkdir -p "$TMPDIR"
echo "nvcc=$(which nvcc)"

# ---------- preflight: imports resolve here, before anything is stopped -----
$HOME/venv70/bin/python -c "
import subprocess, sys
sys.path[:0] = ['$T/src', '$T/packages/tilerl-kernels/src']
import tilerl, tilerl_kernels
print('PREFLIGHT tilerl', tilerl.__file__)
print('PREFLIGHT tilerl_kernels', tilerl_kernels.__file__)
assert '$T' in tilerl.__file__, tilerl.__file__
assert '$T' in tilerl_kernels.__file__, tilerl_kernels.__file__
from tilerl.build import build_engine, build_model
from tilerl.cli import _qwen38_tokenizer
from tilerl.spec import load_draft
print('PREFLIGHT_OK')
" || { echo "FATAL preflight (serve untouched)"; exit 91; }

# ---------- stop the old production by EXACT pid ----------------------------
LIVE=$(cat /proc/$OLD_PID/cmdline 2>/dev/null | tr '\0' ' ')
case "$LIVE" in *"cli serve"*) echo "=== old serve cmdline confirmed";;
  *) echo "FATAL pid $OLD_PID is not a serve ($LIVE)"; exit 90;; esac
echo "=== old serve cwd=$(readlink /proc/$OLD_PID/cwd)"
kill -INT "$OLD_PID"
for i in $(seq 1 120); do
  ST=$(ps -o stat= -p "$OLD_PID" 2>/dev/null | tr -d ' ')
  [ -z "$ST" ] && { echo "=== old serve exited after ${i}s"; break; }
  case "$ST" in Z*) ;; esac   # a zombie has not exited
  sleep 1
done
ps -o stat= -p "$OLD_PID" >/dev/null 2>&1 && { echo "FATAL pid still present"; exit 90; }
LA=$(ps -o ppid= -p "$OLD_PID" 2>/dev/null | tr -d ' ')   # launcher, usually gone
sleep 3; nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---------- prompts ---------------------------------------------------------
$HOME/venv70/bin/python "$D/make_cutover_prompts.py" --home "$HOME" \
  --out "$D/cutover_prompts.jsonl" > "$D/prompts.out" 2>&1
echo "PROMPTS EXIT=$?"; cat "$D/prompts.out"
[ -s "$D/cutover_prompts.jsonl" ] || { echo "FATAL prompts"; exit 90; }
N=$(wc -l < "$D/cutover_prompts.jsonl")

# ---------- ⑤④ correctness + throughput, 3 arms, one subprocess each --------
cd "$T"
echo "===== STAGE 1 THREE ARMS $(date +%T) ====="
echo "# this window is a correctness gate (full-sequence comparison, n=$N), not"
echo "# a speed statistic; the floor of 20 applies to speed estimation."
$HOME/venv70/bin/python -u "$PROBE" \
  --model qwen38-27b --source "$HOME/models/Qwen3.8-27B-NVFP4" \
  --draft "$HOME/mmlu-assets/model_mtp.safetensors" \
  --prompts "$D/cutover_prompts.jsonl" --reference-dir "$D/refs" \
  --expect-tree "$EXPECT_TREE" --out-prefix "$D/cut" --stage 1 \
  --n-prompts "$N" --min-prompts "$N" \
  --min-tokens 4096 --max-tokens 37600 --max-new-tokens 1024 \
  --arms baseline,ref_eager_w2048,graph_w2048 \
  > "$D/probe.out" 2> "$D/probe.err"
echo "PROBE EXIT=$?"
cat "$D/probe.out"

# ---------- ⑤ verdict + ④ short-prompt text sanity --------------------------
python3 - "$D" "$N" <<'PY'
import json, os, sys
d, n = sys.argv[1], int(sys.argv[2])
g = json.load(open(f"{d}/cut_graph_w2048.stage1.json"))
r = json.load(open(f"{d}/cut_ref_eager_w2048.stage1.json"))
b = json.load(open(f"{d}/cut_baseline.stage1.json"))
print("=== ⑤ FULL-SEQUENCE (graph vs forced-eager ref), per prompt")
for p in g["prompts"]:
    print(f"  prompt {p['i']}: {p.get('fullseq')!r} "
          f"n_out={p['n_out']} decode_ticks={p['decode_ticks']} "
          f"graph_ticks={p['graph_ticks']} close={p['close_ticks']}")
print("  failures:", len(g["failures"]))
for f in g["failures"][:10]:
    print("   MISMATCH", f)
print("  unattributed_eager:", len(g.get("unattributed_eager_ticks", [])))
print("=== arms: baseline is the ONLY min8192 arm")
for name, j in (("baseline", b), ("ref_eager_w2048", r), ("graph_w2048", g)):
    t = j.get("throughput") or {}
    print(f"  {name}: min_tokens={j['config']['min_tokens']} "
          f"graph_on={j['config']['built_sparse_graph_on']} "
          f"eff_tok_s={t.get('warm_effective_tok_s')} "
          f"p50_graph={t.get('warm_step_p50_ms_graph')} "
          f"p50_eager={t.get('warm_step_p50_ms_eager')}")
PY

echo "=== ④ short/mid prompt text (first 120 chars, from the graph arm) ==="
$HOME/venv70/bin/python - "$D" "$N" <<'PY'
import json, os, sys
sys.path[:0] = [os.environ["PYTHONPATH"].split(":")[0],
                os.environ["PYTHONPATH"].split(":")[1]]
from tilerl.cli import _qwen38_tokenizer
tok = _qwen38_tokenizer()
d, n = sys.argv[1], int(sys.argv[2])
man = json.load(open(f"{d}/cutover_prompts.jsonl.manifest.json"))["prompts"]
pp = f"{d}/cut_pp"
for m in man:
    i = m["i"]
    for arm in ("graph_w2048", "baseline"):
        p = f"{pp}/{arm}_{i:03d}.json"
        if not os.path.exists(p):
            continue
        out = json.load(open(p))["output"]
        txt = tok.decode(out)
        nuniq = len(set(out))
        print(f"  prompt {i} {m['label']:9s} {m['n_tokens']:6d}tok arm={arm:9s} "
              f"n_out={len(out)} distinct={nuniq} :: {txt[:120]!r}")
PY

echo "=== ④ prefill tick wall, baseline(min8192) vs ref_eager_w2048(min0) ==="
python3 "$D/parse_prefill_ticks.py" \
  "$D/cut_baseline.stage1.err" "$D/cut_ref_eager_w2048.stage1.err" \
  --json "$D/prefill_ticks.json" || echo "PREFILL PARSE rc=$?"

# ---------- start the $M service (min0) -------------------------------------
sed -e "s|--sparse-min-tokens 8192 ||" \
    -e "s|cd \$HOME/tilerl-v100-sse|cd $T|" \
    -e "s|PYTHONPATH=\$HOME/tilerl-v100-sse/src:\$HOME/tilerl-v100-sse/packages/tilerl-kernels/src|PYTHONPATH=$T/src:$T/packages/tilerl-kernels/src|" \
    "$HOME/run_serve_prod.sh" > "$HOME/run_serve_prod.new.sh"
chmod +x "$HOME/run_serve_prod.new.sh"
echo "=== new launcher (diff vs old) ==="
diff "$HOME/run_serve_prod.sh" "$HOME/run_serve_prod.new.sh"; echo "diff rc=$? (1 = differs as intended)"
# Both substitutions must have landed: no old tree path left, no min-tokens left.
grep -n "tilerl-v100-sse" "$HOME/run_serve_prod.new.sh" && { echo "FATAL old tree path remains"; exit 90; }
grep -n "sparse-min-tokens" "$HOME/run_serve_prod.new.sh" && { echo "FATAL min-tokens remains"; exit 90; }
grep -n "cd $T" "$HOME/run_serve_prod.new.sh" || { echo "FATAL new tree not cd'd"; exit 90; }
grep -n "PYTHONPATH=$T/src" "$HOME/run_serve_prod.new.sh" || { echo "FATAL new PYTHONPATH not set"; exit 90; }

echo "===== START SERVICE $(date +%T) ====="
# No `| head -c` anywhere on this redirection: SIGPIPE would kill production.
nohup bash "$HOME/run_serve_prod.new.sh" > "$D/serve-$TS.log" 2>&1 &
echo "=== first log lines:"
for i in $(seq 1 30); do [ -s "$D/serve-$TS.log" ] && break; sleep 2; done
head -5 "$D/serve-$TS.log"

H=""
for i in $(seq 1 90); do
  H=$(curl -s -m 3 http://127.0.0.1:8000/health 2>/dev/null)
  echo "$H" | grep -q '"status":"ok"' && { echo "HEALTH_OK"; break; }
  sleep 5
done
NEW=$(pgrep -f "cli serve" | head -1)
echo "=== ① restored pid=$NEW"
# ② no auto-disabled warning anywhere in the service log
if grep -q "sparse decode graph auto-disabled" "$D/serve-$TS.log"; then
  echo "② RED: auto-disabled present"; else echo "② GREEN: no auto-disabled"; fi
echo "$H" | python3 -c "import sys,json;d=json.load(sys.stdin);s=d['stats'];print('① health',d['model'],'graph',s['decode_graph'],'blocks',s['blocks_total'],'slots',s['slots_total'])"
curl -s -m 60 -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen38-27b","messages":[{"role":"user","content":"reply with the single word ok"}],"temperature":0,"max_tokens":4,"enable_thinking":false}' \
  | python3 -c "import sys,json;print('① CHAT200',json.load(sys.stdin)['choices'][0]['message']['content'])"

# ---------- ③ one real 37.6k prompt over HTTP -------------------------------
$HOME/venv70/bin/python - "$D" <<'PY'
import json, os, sys, time, urllib.request
sys.path[:0] = [os.environ["PYTHONPATH"].split(":")[0],
                os.environ["PYTHONPATH"].split(":")[1]]
from tilerl.cli import _qwen38_tokenizer
tok = _qwen38_tokenizer()
d = sys.argv[1]
man = json.load(open(f"{d}/cutover_prompts.jsonl.manifest.json"))["prompts"]
ids = json.loads(open(f"{d}/cutover_prompts.jsonl").readline())["input_ids"]
text = tok.decode(ids)
base = "http://127.0.0.1:8000"

def health():
    with urllib.request.urlopen(f"{base}/health", timeout=60) as r:
        return json.load(r)["stats"]

body = json.dumps({"model": "qwen38-27b", "temperature": 0, "max_tokens": 256,
                   "enable_thinking": False,
                   "messages": [{"role": "user", "content": text}]}).encode()
req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
s0 = health(); t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=3600) as r:
    out = json.loads(r.read().decode(), strict=False)
wall = (time.perf_counter() - t0) * 1000
s1 = health()
n = out["usage"]["completion_tokens"]
fwd = max(s1["decode_forwards"] - s0["decode_forwards"], 1)
print(f"③ prompt_tokens {out['usage']['prompt_tokens']} completion {n}")
print(f"③ decode_forwards {fwd} eff_tok_s {1000 * n / wall:.1f} "
      f"(wall {wall:.0f}ms incl prefill+HTTP) end_to_end_tok_s {1000 * n / wall:.1f}")
print(f"③ acceptance {(s1['spec_accepted'] - s0['spec_accepted']) / max(s1['spec_drafted'] - s0['spec_drafted'], 1):.3f}")
print("③ sample:", repr(tok.decode(out["choices"][0]["message"]["content"] if isinstance(out["choices"][0]["message"]["content"], list) else [])[:80]))
PY

echo "=== CUTOVERWIN END $(date) dir=$D pid=$NEW ==="
echo "=== handoff: pid=$NEW, tree=$T (HEAD=$(git -C $T rev-parse --short=8 HEAD))"
