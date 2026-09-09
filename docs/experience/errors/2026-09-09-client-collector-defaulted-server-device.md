# A client-side collector defaulted the server's device

## Context

Running the five CPU-safe collectors for B1 requirements gathering on 2026-09-09,
the three server-client collectors — `bench_chat_cold_warm`,
`bench_chat_interleaved`, `bench_workloads` — emitted rows labeled
`device: {name: "H20"}` from runs against a tiny CPU server. Eight mislabeled
rows were deleted from the store (`chat_turn_wall_s` ×6, `prefix_hits` ×1,
`decode_tok_s` ×1); the store went back to 14 rows.

## Root cause

`benchrec.add_record_args` carried `default_device="H20"` for every collector.
A client-side collector measures a remote server over HTTP and cannot see that
server's device — the default was a value the client cannot know, applied
unconditionally.

The tempting fix is to derive the device from `--target`. That is wrong:
`--target` is the compute target (cpu / sm90), `device_name` is the physical
card, and the client knows neither. A derived value would vary with the flag
and look correctly set, which is harder to catch than a fixed wrong default.

## Fix

`add_record_args` gained `client_side=True`. Client-side collectors get no
`--device-name` default, and `record_common` raises without it:

```
--device-name required: a client cannot see the server's device, and a default
here is a population lie -- a cpu run labeled H20 enters every device-grouped
view and every measured-best comparison
```

The rule is written into the `add_record_args` docstring, the shared entry
point, rather than repeated in the scripts. The three client-side collectors
were flipped to `client_side=True`; engine-direct collectors keep the
torch-derived device default. `--build` already had this shape
(`record_common` raises without it) — the device field now matches it.

Runnable check: `tests/test_bench_gate.py::test_client_side_collector_cannot_default_the_server_device`
— a client-side parser without `--device-name` raises; with it, the record
carries the given name.

## Rule

A mislabeled row is worse than a missing one. A missing row is a gap; a row
labeled `device: H20` from a cpu run enters every device-grouped view,
participates in every measured-best comparison, and becomes the "history" an
H20 row is compared against. It is not a missing datum — it is pollution of a
population.

A client-side collector must not default any field describing the server.
Defaults are only for things the collector itself knows.
