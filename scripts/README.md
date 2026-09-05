# scripts

Durable tools — run from a checkout, documented in their docstrings:

| script | what |
|---|---|
| `bench_harness.py` | the perf gate: decode/prefill/train suites vs `docs/experience/wins/bench-baseline.json` |
| `baseline.py` | pull / commit the snapshot the harness gates against |
| `mmlu.py`, `gsm8k_jsonl.py` | accuracy: MMLU through tileRL or sglang; GSM8K → `tilerl train --data` JSONL |
| `rl_compare.sh` | roadmap P5: same pod, same task, tileRL vs verl+sglang |
| `pod_sync.sh`, `pod.sh`, `pod_fan.sh` | sync this checkout to the H20 pod and run there |
| `hf_reference.py`, `health_probe.py`, `verify_h20_fp4.py` | external ground truth and the 27B verify checks |
| `probe_served_rate.py` | the served rate over HTTP, from /health's forward counters — three clock-based instruments got it wrong |
| `cuda_codegen.py`, `op_parity.py`, `parity_*.py` | kernel codegen inspection and parity gates |

Everything prefixed `probe_`, `diag_`, `bench_<kernel>`, `ab_`, `_` is a
one-off from one investigation; its result lives in `docs/experience/` and
the script is kept only so the entry can be re-run. Do not build on one.

And stop keeping it when that stops being true: **a probe whose entry is
superseded, or whose mechanism a later measurement excludes, is deleted — the
entry is the record, not the script.** Repair its inbound references in the
same commit.

"Nothing references it" is not that test, and grepping the script's name is not
that test either. Three ways a live script looks dead: an entry cites the *log*
it wrote and never names it (`bench_smoke.py` — `/work/bench_smoke.log`); it is
a launcher, which by definition nothing imports (`_pod_*.sh`, which carry the
setsid/CUDA_VISIBLE_DEVICES/redirect combination that took real failures to get
right); or it is the only producer of an artifact something else consumes. Check
the log name and what the script *calls* before deleting it.

A dead-code scanner does not apply that test either. Measured 2026-09-05: a
vulture run at 60% confidence produced 11 symbols, and 10 had live callers —
`rmsnorm_bwd`, `attn_prelude`, `iso_merge`, `is_shared`, `write_block` and
`roles` are called from `tests/`, `_trunk_logits`/`_verify_chains` and `init_tp`
from `scripts/`. Both of its **100%**-confidence hits were `types` in
`__torch_dispatch__`/`__torch_function__` — required positional parameters of a
torch protocol, so deleting them breaks the probe at call time rather than at
import. Confidence scores rank how sure the parse is, not how sure the repo is.
Grep `tests/` and `scripts/` for every candidate before proposing a deletion.
