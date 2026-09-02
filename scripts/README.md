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
| `cuda_codegen.py`, `op_parity.py`, `parity_*.py` | kernel codegen inspection and parity gates |

Everything prefixed `probe_`, `diag_`, `bench_<kernel>`, `ab_`, `_` is a
one-off from one investigation; its result lives in `docs/experience/` and
the script is kept only so the entry can be re-run. Do not build on one.
