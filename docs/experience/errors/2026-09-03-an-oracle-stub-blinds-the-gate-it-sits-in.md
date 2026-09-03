---
question: Why did the block-drafter gate pass under two mutations that break the drafter?
source: tests/test_dflash2.py on the tiny fixture, TILERL_TARGET=cpu
---

# A stub that replaces the component under test makes the gate blind to it

The first gate for the block-drafter path had three assertions: the speculated
engine emits the same tokens as the unspeculated one, `spec_drafted > 0`, and —
in a second arm that monkeypatched `head.path` to return the trunk's own
continuation, so a whole block would commit — `decode_forwards` falls below the
unspeculated arm's.

Four mutations, run against it:

| mutation | caught |
|---|---|
| `_draft_step` dispatch removed | yes — `AttributeError` |
| `aux_layers` dropped from `_run_forward` | yes — shape mismatch |
| anchor off by one (`r.tokens[-2]`) | **no** |
| context cache dropped, only this tick's positions kept | **no** |

## Root cause

Two independent holes, and the second is the one worth writing down.

**Output equality cannot see a bad draft.** A rejected draft costs a trunk row
and never a token, so *every* wrong-draft mutation keeps the output identical.
That was known when the test was written; `decode_forwards` was supposed to be
the assertion that saw it.

**The oracle stub deleted the path it was supposed to police.** The patched
`path` read the answer out of the base arm's completion by absolute position:

```python
def path(hidden, anchor, backend):
    i = at["start"] - len(_PROMPT) + 1
    return [base[i + j] for j in range(hidden.shape[1])]
```

It ignores `hidden` and it ignores `anchor`. So the arm that was meant to force
acceptance also discarded everything upstream of it — the anchor the engine
hands over, the context K/V behind `block_hidden`, the whole draft. Acceptance
stayed high under both mutations, `decode_forwards` stayed low, and the gate
stayed green. The stub was written to make one path reachable and it made three
paths unobservable.

## Fix

A content gate, the shape the NextN head already had
(`test_engine_draft_matches_full_context_draft`): capture what the engine
actually drafted, then recompute it independently from the whole token prefix
and require equality.

```python
for tokens, drafts in seen[:3]:
    ctx, pos = tokens[:-1], np.arange(len(tokens) - 1)
    hid = []
    model.forward(np.array([ctx]), pos, _training_kv(...), backend,
                  hidden_out=hid, aux_layers=taps)
    want = head.draft(torch.cat(hid[:-1], -1), pos, tokens[-1], backend)
    assert drafts == want
```

The oracle arm stays — it is still the only thing that commits a whole block —
and gains the one assertion it was missing, that the anchor it is handed is the
token the trunk committed. `max_num_batched_tokens=4` puts the prompt across two
prefill chunks, which is where the context accounting has a boundary.

Six mutations against the fixed gate, all caught: dispatch removed, `aux_layers`
dropped, anchor off by one, cache dropped, sliding window shortened to 1, block
start one position late. A seventh — restoring the prefill-phase boundary to
`seq_len - 1` — trips the hole check by name, `context K/V has a hole at 3..4`.

One mutation that looked like a control and was not: trimming the context
whenever its width exceeds 0 rather than the sliding window. On the tiny fixture
the window is 6 and the contexts run past it, so "always trim to the last 6" is
what the correct code already does. A mutation that lands on the same behaviour
is not a negative control; it says nothing either way.

## Rule

A stub inserted to make a path reachable must still consume the inputs the real
component consumes, or every mutation upstream of it is invisible. Check what
the stub ignores, and assume nothing above it is covered.

When output equality is insensitive by construction — speculative decoding,
caches, any accept/reject stage — the gate has to compare the intermediate the
stage produces against an independent recomputation. Counting how many forwards
it took is a proxy, and a proxy is only as good as the arm that produces it.
