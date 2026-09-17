---
status: guidance
source: D-batch dead-code pass, 2026-09-17; a name-based scan flagged load-bearing test code
---

# Name-based dead-code scans false-positive on framework-reflected symbols

## Rule

**Zero textual references does not mean dead.** Before deleting a symbol with
no call sites, ask who invokes it: a framework, a test runner, or the interpreter
may dispatch it **by convention name at runtime**, and that dispatch is never a
name in the source. A grep over the repo cannot see a caller that lives outside
the text.

Three classes in the test tree all read "defined once, referenced nowhere", and
all three are load-bearing:

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

## Why grep/reference counting misses all three

The caller is not a token in the repo. It is:

- pytest's fixture registry keyed by the function object collected at import;
- `getattr(self, "visit_" + type(node).__name__)` inside `NodeVisitor`;
- `getattr(handler, "do_" + command)` / the stdlib logging hook.

A scan that counts textual references counts zero for every one and cannot, from
the name alone, tell a reflected hook apart from an orphan.

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

This mirrors the scripts closure audit's own standard
(`scripts/audit_scripts_entrypoints.py`): classify by **enumerated reachability**,
not keyword search, and keep an explicit, reason-carrying registry only for the
invisibility a structural scan cannot model (a tool a person runs by hand). The
same principle extends to tests: the registry of last resort is for genuinely
name-invisible callers, and reflected hooks are not that — they are structurally
recognizable.
