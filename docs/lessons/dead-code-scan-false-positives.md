---
status: guidance
source: D-batch dead-code pass, 2026-09-17; a name-based scan flagged load-bearing test code
---

# Name-based dead-code scans false-positive on reflected symbols and registry/accounting names

## Rule

**Zero textual references does not mean dead, and a name appearing does not mean
it is called.** Two distinct traps:

- A symbol with no call sites may be dispatched **by convention name at runtime**
  by a framework, a test runner, or the interpreter — that dispatch is never a
  name in the source (classes 1–3).
- A kernel/function **name** can appear in the source purely as **data** — a
  registry key or an accounting row — while the thing it names is never invoked
  there. That reference documents *coverage* (often the absence of coverage), not
  a call (classes 4–5).

Before deleting, ask two questions: who invokes it, and is this occurrence a call
or a key? A grep over the repo answers neither when the consumer lives outside the
text or the token is a string.

Five classes in the test tree all look removable under a naive name/call scan,
and all five are load-bearing:

### 1. pytest fixtures — injected by the runner, not called by name

A fixture is invoked by pytest during collection/setup. `autouse=True` fixtures
in particular have **no textual reference at all**: pytest matches the decorated
function to every test in scope by its registration, so the name appears exactly
once (the `def`).

- Example (at-time `tests/test_card_guard.py:18`): `_clean_env`, an
  `@pytest.fixture(autouse=True)` that strips `TILERL_CARD_LEND` /
  `CUDA_VISIBLE_DEVICES` before each test. Deleting it leaves every test
  inheriting a polluted environment; no test names it.

### 2. `ast.NodeVisitor.visit_*` methods — dispatched by the AST walker

`NodeVisitor.visit` looks up `visit_<ClassName>` on `self` with `getattr` and
calls it reflectively. A `visit_If` / `visit_Call` / `visit_ClassDef` override
is never called textually; the generic `visit()` the base class provides is the
caller.

- Example (at-time `tests/test_step_timing.py`): `_Finder.visit_ClassDef`,
  `visit_If`, `visit_Assign` and `_Walk.visit_Call` are the sync-free AST gate.
  A reference scan sees four unused methods; deleting them silently turns the
  gate into a no-op that passes on anything.

### 3. Stdlib/base-class overrides — dispatched by the framework

Subclass hooks such as `http.server.BaseHTTPRequestHandler.do_GET` and
`log_message`, `argparse` actions, `unittest.TestCase.setUp`/`tearDown`, context
manager/dunder methods, etc., are resolved by name from the base class or
protocol machinery. Overrides exist specifically to be called reflectively.

- Example (at-time `tests/test_serve_liveness.py`): the `_503`
  `BaseHTTPRequestHandler` subclass overrides `do_GET` (returns 503) and
  `log_message` (silences the server). Neither is called in the file; the HTTP
  server invokes them on each request.

### 4. Registry membership queries — the name documents that a kernel is (or is NOT) in a cell

A test can inspect a kernel registry as data — `"name" in cell`, `_resolve(...)[
"name"]`, `for name, fn in _REGISTRY.items()` — to assert which backends ship a
kernel. The named kernel is never invoked in the test, and the most valuable of
these assert **absence**: "the CPU cell does not register this kernel, so a CPU
parity gate cannot see the defect". Deleting the string deletes the record of
the coverage hole.

- Examples (at-time):
  - `tests/test_attn_prelude_oracle.py:43`
    `test_the_cpu_cell_cannot_observe_the_preludes_extra_rounding`:
    `rmsnorm_fused` appears only in `assert "rmsnorm_fused" not in cpu`,
    `_resolve("fp4","cpu")`, and `sm90["rmsnorm_fused"].__name__`. It asserts
    the bf16-output norm is sm90-only, which is why the prelude's extra rounding
    needs a device parity gate.
  - `tests/test_fused_projections_parity.py:77`
    `test_attn_prep_has_no_cpu_twin_...`: calls only the `RefBackend.attn_prep`
    stub (`... is None`); the real fused `attn_prep` kernel is never dispatched
    on CPU, and the test exists to say so.
  - `tests/test_rmsnorm_f32_tape.py:29`
    `test_every_cell_can_actually_deliver_an_f32_norm_output`: walks
    `_REGISTRY.items()` checking whether each cell provides `rmsnorm_fused_f32`.
    A pure static-table check; the kernel is not run here.

### 5. Accounting/row-name strings — the name labels a roofline or resolution row

A kernel name can be a literal in an expected-set against a cost table's
`name` field, or the expected value of a face→kernel resolution over a
`FakeBackend`. It labels bookkeeping, not an execution.

- Examples (at-time):
  - `tests/test_kernel_cost.py:263`
    `test_tick_rows_cover_the_launched_set_...`:
    `{"paged_attention_decode","gdn_decode_fused",...} <= {r["name"] for r in
    tick_rows(...)}` asserts the roofline table names the launched kernels; the
    strings are row labels.
  - `tests/test_calibration.py:247`
    `test_resolve_row_kernel_is_the_declared_kernel_not_linear`:
    `"linear_fp4"`/`"linear_fp8"` are the *expected* resolution results; the
    backend is a `FakeBackend` of sentinel lambdas, so nothing real executes.

## Why grep/reference counting misses all five

For classes 1–3 the caller is not a token in the repo. It is:

- pytest's fixture registry keyed by the function object collected at import;
- `getattr(self, "visit_" + type(node).__name__)` inside `NodeVisitor`;
- `getattr(handler, "do_" + command)` / the stdlib logging hook.

A scan that counts textual references counts zero for every one and cannot, from
the name alone, tell a reflected hook apart from an orphan.

For classes 4–5 the trap is the inverse: the name **does** appear, but as a
string the test compares against a registry or a cost table, not as a callee.
A scan that looks for "is this name referenced?" reports it as used and stops
thinking; a scan that looks for "is the kernel actually invoked?" still sees no
call. The occurrence is an assertion *about* dispatch (which cell ships it,
which row labels it), and in the absence-asserting cases it is the only place
the coverage gap is recorded.

## How a dead-code audit should exempt them

Exempt by **structure (AST), not by an allow-list of literal filenames**:

- **Fixtures**: parse the tree; a `FunctionDef` (or async def) carrying a
  `@pytest.fixture` decorator — with or without `autouse` — is registered. An
  autouse fixture needs no reference; a named fixture is alive if any test in its
  scope requests the parameter name.
- **Visitor hooks**: a method whose name matches `visit_<Name>` on a class that
  subclasses `ast.NodeVisitor` (or defines `visit_*` dispatch) is an override of
  a known reflective hook — exempt the set of `NodeVisitor` hook names within
  such a class.
- **Base-class overrides**: for a class subclassing an imported framework/stdlib
  base, treat any method that exists on the base class as a dispatched override
  (`do_GET`, `log_message`, `setUp`, `__enter__`, …). Resolve the base class's
  method set rather than hard-coding names per file.
- **Registry/accounting names**: distinguish a *call* from a *string key*. A
  name literal that is the subject of a membership/equality assertion
  (`"k" in cell`, `cell["k"].__name__`, an expected-set over a table's `name`
  field), iterated from a registry (`_REGISTRY.items()`), or compared against a
  `FakeBackend` resolution is data about coverage, not an invocation. Exempt
  string literals used in these assertion shapes; do not delete one whose
  message text asserts absence (it documents a gate the CPU cell cannot run).

This mirrors the scripts closure audit's own standard
(`scripts/audit_scripts_entrypoints.py`): classify by **enumerated reachability**,
not keyword search, and keep an explicit, reason-carrying registry only for the
invisibility a structural scan cannot model (a tool a person runs by hand). The
same principle extends to tests: the registry of last resort is for genuinely
name-invisible callers, and reflected hooks and registry-keyed coverage
assertions are not that — they are structurally recognizable.
