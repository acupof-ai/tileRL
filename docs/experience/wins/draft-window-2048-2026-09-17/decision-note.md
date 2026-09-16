# Draft trailing-window W decision table (16k/32k) — V100 sm70, 2026-09-17

Probe scripts/probe_draft_window_sweep.py (#696 merge ab054700, probe
535bea76), in-process eager engine, sparse_k=128, d1, NoPrefixStore,
wikitext-103 test (real corpus, disjoint spans), 96 out. n=18 at 16k, n=9
at 32k (corpus-limited: test split 297054 tokens; #696 adaptive n).
9k is n=30 from the earlier full-table run, recovered from stdout after
that run hit the corpus assert before its JSON flush.

The probe is eager (sm70 graph auto-disabled), and the draft head is always
eager on the served path too, so draft_ms is the same kernel path as the
live hybrid serve. Native tok/s is the eager probe's own (not served); at
32k n=9 it is non-monotonic (4.71/4.65/8.41/4.48 vs W0 6.36) and
uncorrelated with draft_ms/accept/cold_fill, i.e. scheduler/JIT noise — the
decision column is modeled_served: tick_ms = #665 live-tick medians
trunk_model(151/150/163 at 9k/16k/32k) + draft_ms(W) + overhead(6/7/14);
modeled_served = accept_len / tick_ms x1000. Cross-validates within ~0.3
tok/s of native at W0 on 16k.

| len | W | draft_ms | accept | accept_len | modeled tick | modeled served | vs W0 | n |
|---|---|---|---|---|---|---|---|---|
| 9k | 0 | 33.7 | .757 | 1.751 | 190.7 | 9.18 | — | 30 |
| 9k | 1024 | 9.2 | .721 | 1.713 | 166.2 | 10.31 | +12.3% | 30 |
| 9k | **2048** | 11.2 | .743 | 1.737 | 168.2 | **10.33** | +12.5% | 30 |
| 9k | 4096 | 15.5 | .743 | 1.735 | 172.5 | 10.06 | +9.6% | 30 |
| 9k | 8192 | 30.0 | .746 | 1.736 | 187.0 | 9.28 | +1.1% | 30 |
| 16k | 0 | 59.0 | .755 | 1.744 | 216.0 | 8.08 | — | 18 |
| 16k | 1024 | 9.6 | .717 | 1.707 | 166.6 | 10.25 | +26.9% | 18 |
| 16k | **2048** | 11.8 | .729 | 1.721 | 168.8 | 10.19 | +26.2% | 18 |
| 16k | 4096 | 16.6 | .749 | 1.740 | 173.6 | 10.02 | +24.1% | 18 |
| 16k | 8192 | 30.0 | .749 | 1.740 | 187.0 | 9.30 | +15.2% | 18 |
| 32k | 0 | 117.4 | .754 | 1.746 | 294.4 | 5.93 | — | 9 |
| 32k | 1024 | 9.8 | .699 | 1.691 | 186.8 | 9.05 | +52.7% | 9 |
| 32k | **2048** | 12.1 | .723 | 1.715 | 189.1 | **9.07** | +52.9% | 9 |
| 32k | 4096 | 16.7 | .734 | 1.726 | 193.7 | 8.91 | +50.3% | 9 |
| 32k | 8192 | 30.5 | .725 | 1.717 | 207.5 | 8.28 | +39.6% | 9 |

All arms proof=ok (W0 never engages; W>0 first-page>0, windowed_seq_len
tracks W). No failures.

## Decision — W=2048 is a MARGIN pick, not a statistically significant winner

W1024 and W2048 are NOT statistically separable at 32k n=9 (~500 drafted
samples/arm, acceptance 95% interval +-0.04; the .699 vs .723 gap of 2.4
points is inside it). W2048 is chosen on safety margin, not proven
superiority: it keeps +2.4 acceptance points over W1024 for only +2.3 ms
draft, it is the optimum at 9k, within noise of W1024 at 16k, and one value
covers all three lengths (no length-dependent knob). Beyond 2k acceptance
plateaus (.723/.734/.725) while draft cost rises 12->30 ms, so 4k/8k are
strictly worse.

Acceptance CI derivation: d1 acceptance per draft tick is Bernoulli; 9
prompts x 96 out / accept_len ~1.72 ~= 56 ticks/prompt ~= 500 drafted-token
samples/arm; p=.70 SE=sqrt(.70*.30/509)=0.020, 95% +-2SE=+-0.041; p=.75
+-0.039. The W1024(.699) vs W2048(.723) gap 0.024 < +-0.04 -> inseparable.

Net at W=2048: sparse served decode modeled 9.18->10.33 at 9k (+12.5%),
8.08->10.19 at 16k (+26%), 5.93->9.07 at 32k (+53%); the 3.6 us/token
full-prefix draft scan (117 ms at 32k) drops to 12 ms (0.10x), acceptance
down only .03.

Caveats / follow-ups:
- 32k n=9 CI +-0.04: W1024/2048 indistinguishable here. BEFORE treating the
  permanent default as settled, re-run 32k n>=30 on wikitext-103 TRAIN split
  to confirm W1024/2048 do not diverge (train parquet availability/scp to
  V100 to be solved then).
- Modeled_served is a live-tick extrapolation, not a live measurement;
  confirm ~9 tok/s on the real hybrid serve after the default-flip deploys
  (perf1 GREEN).
- The probe window is READ-only (draft still writes/retains full dense KV);
  the product sliding-window default lands with the #684 path.
- Native probe tok/s at 32k is noise under n=9 and must not be cited.

Data: resume-16k-32k.json (this run's probe aggregates, n=18/9) and
recovered-9k.json (the earlier n=30 full-table 9k rows, recovered from
stdout). Collected on V100 by perf1; modeled table cross-confirmed.
