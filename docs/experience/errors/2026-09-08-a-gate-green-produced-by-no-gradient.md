# Three defects on a path that had never been executed, and one of them was my gate's green

## Context

`scripts/steps_to_reward.py` is the steps-to-target instrument. Its defaults measure a CPU
fixture that is VOID for the 2.7x question, so the real run needs three overrides: a real
checkpoint, GSM8K correctness as the reward, and a data file. All three were parameterized
in #307 and none had been executed — the pod was frozen, so the flags existed and their
default values kept the new path dark.

Driving them before the run, rather than at the start of it, found three defects.

## Root cause

**1. `--model <path>` silently built a random tiny.** `_build_model` dispatches on the NAME:
`cli.py:56` matches `qwen38-27b`, and `:71` falls through to `config.tiny()` for everything
else. Passing the real checkpoint directory returned `vocab 320 hidden 64 layers 2` against
the checkpoint's 248320/5120/64. It does not raise. A full run would have produced a table
that reads entirely like a 27B measurement. This is worse than an ImportError, which at
least stops.

**2. The tokenizer line called the wrong function.** `get_tokenizer` takes a hub id or a
directory, not a model name: `get_tokenizer("qwen38-27b")` raises
`RepositoryNotFoundError 401`. The name resolves only through `cli._qwen38_tokenizer`. The
gsm8k branch had been written from memory and never run.

**3. My own validity gate printed a green produced by the absence of the thing it measures.**
On tiny + GSM8K the run printed `Sigma drift 0.00%` and
`OK: the free arm holds the spectrum, so the paper's condition is reproduced`. The mechanism:
the model scores 0 on every problem, so every reward in a group is equal, so GRPO's
within-group normalization yields an all-zero advantage — `tied groups 100.0%`, read off
`grpo_loop`'s `h[3]`. No step changed a weight. Σ therefore *cannot* have moved, and the gate
read that arithmetic as evidence that the spectrum was preserved.

The gate's job is to decide whether this fixture reproduces the paper's condition. A
100%-tied run has not tested that condition. Reporting OK is the gate answering a question it
never asked.

## Fix

`MODELS = ("tiny", "tiny-agent", "qwen38-27b")` as argparse `choices`, so a path is refused
at parse time; a local checkpoint arrives through `TILERL_QWEN38_SOURCE`, which
`--model qwen38-27b` already reads. The qwen branch calls `cli._qwen38_tokenizer()`.
`rl_arm` returns `tied`, `main` prints it per arm, a VOID 0 fires above 99% (before the other
two voids, which it invalidates), and `sigma_verdict(tied, ada_max)` has a third outcome:
`NOT TESTED: no gradient was applied, so a 0% drift says nothing` — neither OK nor VOID,
because both would be verdicts on an untested condition.

**One mutant survived the first round, and it was the assert for defect 2.** The assert was
`"_qwen38_tokenizer" in inspect.getsource(make_reward)` — and the comment I had just written
above the call contains that name, so reverting the call to `get_tokenizer(a.model)` passed.
The probe matched the comment, not the behaviour. Replaced with a monkeypatch of both
functions that records **which one was actually called**; mutant 2 then dies with
`got ['hub:qwen38-27b']`. The other two mutants die directly (path accepted, no-gradient run
read as OK). `__pycache__` cleared between each. Negative control: the token-rate arm runs at
`tied 40%` and still reaches a VOID verdict, so the new void does not swallow every run.

## Rule

A flag whose default keeps its path dark is an untested path, and the defect it hides is
usually worse than a crash: three of them here, and the two that do not raise both produce a
finished-looking table. Drive every branch a real run will take, before the run.

And when a gate reports a condition satisfied, check whether the mechanism that would satisfy
it ran at all. A zero can mean "held steady" or "never moved because nothing happened", and a
gate that cannot separate those two reports the second as the first. Ask what produced the
green, not whether it is green — an assert that greps source can be satisfied by a comment.
