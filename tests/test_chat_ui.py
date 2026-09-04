"""The served chat page: 480 lines of CSS and a streaming state machine, gated.

Nothing rendered ``/`` before this file. `_CHAT_UI` is a single string constant,
so every defect in it is invisible to the test suite -- two sessions in a row
reviewed a version of this page that had already been replaced, because there
was no gate to fail when it moved.

These are structural assertions, not a snapshot: a snapshot of a 20 KB string
fails on every edit and teaches nothing. Each test names one property the page
must keep and fails only when that property breaks.

Two of them are about the inline JS rather than the CSS. `addThinking` shipped
called-but-never-defined and no gate could see it: the whole suite asserts on
SSE bytes, and `_CHAT_UI` is 21 KB of JavaScript that Python only measures the
length of. `node --check` does NOT close this -- verified: it reports SYNTAX_OK
on `function a(){ return undefinedFn(1); }`, because an undefined call is a
runtime error. The resolver below needs no JS runtime and so always runs in CI.

# ponytail: undefined calls only, stub-DOM execution when the page grows
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from tilerl.server import _CHAT_UI

#: Statement keywords that a naive "identifier followed by (" also matches.
_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "typeof", "function",
    "await", "new", "do", "else", "delete", "void", "in", "of",
}

#: Globals the browser provides. Anything called bare and absent here has to be defined
#: in the script itself, which is exactly the addThinking case.
_BROWSER_GLOBALS = {
    "AbortController", "Error", "String", "Number", "Boolean", "Array", "Object",
    "JSON", "Math", "Date", "Promise", "Map", "Set", "RegExp", "TextDecoder",
    "TextEncoder", "URL", "URLSearchParams", "FormData", "Headers", "Request",
    "Response", "fetch", "setTimeout", "clearTimeout", "setInterval",
    "clearInterval", "requestAnimationFrame", "cancelAnimationFrame", "alert",
    "confirm", "parseInt", "parseFloat", "isNaN", "encodeURIComponent",
    "decodeURIComponent", "structuredClone", "queueMicrotask", "btoa", "atob",
}


def _css() -> str:
    """The chat page's <style> block alone.

    `server.py` holds two pages; slicing from the file's first `<style>` picks
    up the landing page instead, whose spacing is its own concern. Slice from
    `_CHAT_UI` itself so this can only ever describe the chat page.
    """
    head = _CHAT_UI.index("<style>")
    return _CHAT_UI[head : _CHAT_UI.index("</style>", head)]


def _script(html: str) -> str:
    assert "<script>" in html, "the page has no inline script"
    return html.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def _code_only(js: str) -> str:
    """Blank out string literals, template literals and comments.

    Without this, UI copy triggers false positives: `"read/write files  (Enter to send)"`
    reads as a call to `files`. Measured -- it reported `files` and `tilerl` alongside the
    real `addThinking`. Replaced with spaces, not deleted, so nothing new becomes adjacent.
    """
    pattern = (
        r'"(?:[^"\\\n]|\\.)*"'      # double-quoted
        r"|'(?:[^'\\\n]|\\.)*'"     # single-quoted
        r"|`(?:[^`\\]|\\.)*`"       # template literal
        r"|//[^\n]*"                # line comment
        r"|/\*.*?\*/"               # block comment
    )
    return re.sub(pattern, lambda m: " " * len(m.group(0)), js, flags=re.S)


def _unresolved(js: str) -> set[str]:
    js = _code_only(js)
    # A bare call: identifier + "(" with no preceding '.' (which would make it a method)
    # and no preceding word character (which would make it a suffix of a longer name).
    bare = set(re.findall(r"(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(", js))
    defined = set(re.findall(r"function\s+([A-Za-z_$][\w$]*)", js))
    defined |= set(re.findall(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=", js))
    # Arrow-function parameters, so a callback's own name does not read as unresolved.
    defined |= set(re.findall(r"\(?\b([A-Za-z_$][\w$]*)\)?\s*=>", js))
    # Declared parameters: a name a function receives and then calls, e.g.
    # `readSSE(resp, onFrame)` calling `onFrame(...)`.
    for params in re.findall(r"function\s*[\w$]*\s*\(([^)]*)\)", js):
        defined |= {p.strip().lstrip("...").split("=")[0].strip() for p in params.split(",")}
    return bare - _KEYWORDS - _BROWSER_GLOBALS - defined - {""}


def test_every_bare_call_in_the_page_js_resolves():
    unresolved = _unresolved(_script(_CHAT_UI))
    assert not unresolved, (
        f"the page calls {sorted(unresolved)} and nothing defines them. This is the "
        f"addThinking failure: the reply dies mid-stream with a ReferenceError, and no "
        f"assertion on the SSE bytes can see it because the server is correct."
    )


def test_the_check_catches_an_undefined_call():
    """Negative control: the gate above is worthless if it cannot fail.

    Without this, a regex that quietly matches nothing would report a clean page forever.
    """
    assert _unresolved("function go() { return addThinking(bubble); }") == {"addThinking"}
    # A method call on an object is NOT a bare call, so it must not be flagged.
    assert _unresolved("x.addThinking(1); [].push(2);") == set()
    # UI copy must not read as code. Measured: before literals were stripped, the check
    # reported `files` and `tilerl` from a placeholder string alongside the real bug.
    assert _unresolved('let p = "read/write files  (Enter to send)";') == set()
    assert _unresolved("// call ghost(1) in a comment\nlet a = 1;") == set()


def test_the_index_route_serves_the_page():
    """`/` returns the page as HTML. The route is one line and nothing covered it."""
    from tilerl_kernels.backend import get_backend

    from tilerl.config import tiny
    from tilerl.engine import build_engine
    from tilerl.model import build_random
    from tilerl.server import create_app
    from tilerl.tokenizer import get_tokenizer

    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=3), get_backend(),
                          num_blocks=64, num_slots=2, max_batch=2, max_total_tokens=512)
    app = create_app(engine, get_tokenizer(None), model_name="tiny")
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>" in r.text and "</html>" in r.text


def test_every_gutter_offset_comes_from_one_token():
    """`.turn` is a label column plus a gap, and `.note`/`.ev` sit at that same
    offset. Three rules used to hardcode 80px with nothing tying them to the
    62+18 they mirror, so changing the label column silently desynced them."""
    css = _css()
    assert "--gutter:" in css, "the label-column offset is not a token"
    # No RULE may spell the composed offset as a literal again. Comments are
    # exempt; the sum is worth naming in prose where the token is defined.
    bad = [ln.strip() for ln in css.splitlines()
           if "80px" in ln and "--gutter" not in ln and "max-width" not in ln
           and not ln.lstrip().startswith(("/*", "*", "the "))]
    assert not bad, f"gutter offset hardcoded instead of var(--gutter): {bad}"


def test_spacing_is_spent_from_a_scale():
    """The palette, type, shadow and radius are tokenized; spacing was not, and
    23 hand-picked pixel values is what "the margins between components are
    wrong" looks like. Tokens do not have to cover every value -- asymmetric
    optical padding is real -- but the common ones must come from the scale."""
    css = _css()
    tokens = set(re.findall(r"--s-[\w-]+", css))
    assert len(tokens) >= 5, f"expected a spacing scale in :root, found {tokens}"
    # The scale must actually be spent, not merely declared.
    uses = len(re.findall(r"var\(--s-", css))
    assert uses >= 20, f"spacing scale declared but barely used ({uses} uses)"


def test_no_style_rules_for_components_that_cannot_render():
    """#60 removed the tab strip and every event kind but `error`; their CSS
    stayed. Dead spacing rules are the hardest kind to review -- they look like
    layout decisions for something you cannot find on the page."""
    css, page = _css(), _CHAT_UI
    for cls in (".tab", ".ev.thought", ".ev.action", ".ev.observation", ".ev.answer"):
        if cls in css:
            marker = f'class="{cls.lstrip(".").replace(".", " ")}'
            kind = cls.rsplit(".", 1)[-1]
            reachable = marker in page or f'addEvent("{kind}"' in page
            assert reachable, f"{cls} is styled but nothing can render it"


def test_pending_survives_reduced_motion():
    """The caret and the header dot are the only two pending affordances and
    both are animations, so `prefers-reduced-motion: reduce` -- which kills
    `animation` on `*, *::after` -- used to leave the wait almost unsignalled."""
    css = _css()
    i = css.index("@media (prefers-reduced-motion: reduce)")
    block = css[i : css.index("}\n  }", i) + 5] if "}\n  }" in css[i:] else css[i:]
    assert "streaming" in block, (
        "the reduced-motion block strips the caret's animation without giving "
        "pending a static cue"
    )


def test_the_stream_marks_and_unmarks_the_pending_bubble():
    """The caret is driven by one class. If it is added and never removed, every
    finished turn keeps a blinking cursor; if never added, the wait is silent."""
    assert '.classList.add("streaming")' in _CHAT_UI
    assert '.classList.remove("streaming")' in _CHAT_UI
