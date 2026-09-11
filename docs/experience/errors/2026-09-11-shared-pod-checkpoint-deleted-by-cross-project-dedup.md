# The 27B checkpoint was deleted from the shared pod — 2026-09-11

## Context

Two separate disappearances of the 27B checkpoint on the H20 pod's
`sglang-test` container, on top of the 2026-09-10 *wrong-machine* false alarm
([2026-09-10-the-pod-is-wiped-was-the-wrong-machine.md](2026-09-10-the-pod-is-wiped-was-the-wrong-machine.md)
— that time the checkpoint was intact; the reader was on the host, not the
container). These two are real deletions inside the container, verified
through `crictl exec`.

## First occurrence (2026-09-10)

`/work/Qwen3.8-27B-NVFP4` was gone from inside `sglang-test`; the shared
bench store and the calibration/recall work depended on it. Recorded in the
card-recall notes; the checkpoint later reappeared/restored and was used for
the 09-11 P1 runs (tree 5e2e5794 loaded it successfully).

## Second occurrence (2026-09-11 ~08:00–09:18 UTC)

The indexer recall science run was staged: corpus prep ran at 07:55 and
**successfully loaded `/work/Qwen3.8-27B-NVFP4/tokenizer.json`**. At 09:16 a
follow-up prep failed on a missing tokenizer; by 09:18 the whole directory
was gone:

- `ls /work/Qwen3.8*` empty inside `sglang-test`; nothing on `/mnt/data02`,
  `/mnt/nvme_probe`, `/data00` (container-visible), or `/root`.
- `/work` itself intact: ext4 on `/dev/vda2`, ~754 GB free (the 22.6 GB
  model freed the space — a deletion, not a dropped mount).
- Survived: the session trees under `/work/tilerl-*`, `/work/tl013`, the
  corpora and run dirs. Only the checkpoint directory was removed.

## Root cause

A cross-project process (aupai-2b) deleted `/work/Qwen3.8-27B-NVFP4` as a
"duplicate" of the host's `/data00` NVMe copy. `/data00` is **not mounted in
`sglang-test`**, so inside the container the deleted copy was the only one.
The deletion was correct from the deleter's host view and destructive from
every container consumer's view — a shared-host, separate-mount footgun: a
file that looks redundant on the host can be the sole visible copy in a
container that mounts only `/work`.

## Fix / rule

- tileRL now keeps an owned checkpoint at a tileRL-namespaced path,
  `/work/tilerl-ckpt/Qwen3.8-27B-NVFP4`, copied from the host `/data00`
  rather than relying on a shared, un-namespaced top-level dir another project
  may judge redundant. Point `TILERL_QWEN38_SOURCE` / `TILERL_27B_CKPT` there.
  The copy is bit-exact with the deleted `/work` file: sha256
  `c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05`
  (`/work/tilerl-ckpt/model.sha256`), identical to the hash aupai-2b recorded
  for the deleted copy.
- Before deleting a "duplicate" on a shared host, verify the file is visible
  at the SAME path in every container that mounts the volume — redundancy on
  the host is not redundancy inside a container with a narrower mount set.
- A disappearance that frees the file's space (`df` shows the delta) is a
  deletion; a disappearance with unchanged free space and missing mounts is a
  mount/namespace issue. Check `df` and the mount table before concluding.

## What was blocked

The 27B indexer recall run (unit D, PR #512) could not smoke or run without
the checkpoint; all CPU-gated code and the prepared 8k/16k/32k Chinese-wiki
corpus were ready and unaffected. No fabricated path was substituted.

## Known tail

The two launchers every pod run sources (`scripts/pod_env.sh`,
`scripts/pod_run.sh`) now default `TILERL_QWEN38_SOURCE` to the tileRL-owned
path. Other scripts still spell the old `/work/Qwen3.8-27B-NVFP4` path in one
of two ways, and are deliberately left on a follow-up rather than swept into
this docs PR:

- Dated probes whose path is inside a recorded invocation/usage string
  (~50 files) — historical record, like the wins/errors entries; changing them
  falsifies how those past probes ran.
- A handful of standalone probes with their OWN executable default
  (`probe_spin_cost_27b.py`, `verify_h20_fp4.py`, `capture_determinism.py`,
  `fwd_determinism.py`, `poison_pool_determinism.py`, `probe_kv_fp8_27b.py`,
  `rl_compare.sh`); these are not sourced by `pod_run`, so they do not affect
  a standard launch, but they should be repointed the next time each is run.
  `serve_v100.sh` correctly keeps a V100-box-local path.
