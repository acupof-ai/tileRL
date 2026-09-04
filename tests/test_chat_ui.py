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
SSE bytes, and `_CHAT_UI` is a single Python string that Python only measures the
length of. `node --check` does NOT close this -- verified: it reports SYNTAX_OK
on `function a(){ return undefinedFn(1); }`, because an undefined call is a
runtime error. The resolver below needs no JS runtime and so always runs in CI.

Which bounds what these eight tests see, so state it in measured bytes rather than
call the page "21 KB of JavaScript": of 19.6 KB, only **6.8 KB is script**. The
other **12.9 KB is CSS and markup** -- a 1.91:1 split -- and the resolver never
parses it. Measured by mutating the real source: a dangling *call* is caught, a
dangling *route* (a string literal) is not, a dangling *button* (markup) is not,
and the CSS gates catch a spacing regression but nothing about behaviour. Sizes
drift with every edit to the page; the split is the part worth keeping.

# ponytail: undefined calls only, stub-DOM execution when the page grows
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap

import pytest
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


def test_the_reasoning_split_handles_a_reply_that_starts_inside_think():
    """The stream carries only `</think>`, and the page has to fold on that.

    The checkpoint's template ends the prompt with "<think>\\n", so generation begins
    inside the block: measured against the served V100, a 300-token reply contained
    `</think>` and no `<think>`. The first version keyed on the opening tag, so the
    fold never fired and the reasoning printed inline as prose -- and no gate saw it,
    because every assertion here was about markup and CSS.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; splitThink's behaviour cannot be exercised")
    js = _script(_CHAT_UI)
    start = js.index("function splitThink")
    fn = js[start : js.index("\nfunction ", start + 1)]

    # (raw, inside) -> (reasoning, answer). `inside` is the page's THINK flag.
    cases = [
        # What the server actually sends: no open tag, close tag mid-reply.
        ["why\n</think>\nthe answer", True, "why\n", "\nthe answer"],
        # Mid-stream, before the close tag arrives: all reasoning, no answer yet.
        ["thinking about it", True, "thinking about it", ""],
        # Thinking off: no tags at all, so all of it is the answer.
        ["just the answer", False, "", "just the answer"],
        # A reply that does carry both tags (backfilled history, or a paste).
        ["<think>r</think>a", False, "r", "a"],
        # An unclosed open tag is still reasoning, not an answer.
        ["<think>r only", False, "r only", ""],
        # A close tag with nothing before it: empty reasoning, not a dropped answer.
        ["</think>a", True, "", "a"],
    ]
    harness = fn + "\nconst C = " + json.dumps(cases) + ";\n" + textwrap.dedent("""
        const bad = [];
        for (const [raw, inside, wantR, wantA] of C) {
          const [r, a] = splitThink(raw, inside);
          if (r !== wantR || a !== wantA) bad.push([raw, inside, [r, a], [wantR, wantA]]);
        }
        console.log(JSON.stringify(bad));
    """)
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"splitThink threw: {r.stderr.strip()[:400]}"
    bad = json.loads(r.stdout.strip().splitlines()[-1])
    assert not bad, f"splitThink is wrong on: {bad}"


def test_the_index_route_serves_the_page():
    """`/` returns the CHAT page, and the landing page is still reachable at /about.

    Asserting 200 + text/html + <title> cannot tell the two pages apart -- both satisfy
    all three -- so the route swap would have been invisible. Key on the composer, which
    only the chat page has.
    """
    from tilerl_kernels.backend import get_backend

    from tilerl.config import tiny
    from tilerl.engine import build_engine
    from tilerl.model import build_random
    from tilerl.server import create_app
    from tilerl.tokenizer import get_tokenizer

    cfg = tiny()
    engine = build_engine(cfg, build_random(cfg, seed=3), get_backend(),
                          num_blocks=64, num_slots=2, max_batch=2, max_total_tokens=512)
    client = TestClient(create_app(engine, get_tokenizer(None), model_name="tiny"))
    for route in ("/", "/chat"):
        r = client.get(route)
        assert r.status_code == 200, route
        assert r.headers["content-type"].startswith("text/html"), route
        assert "<title>" in r.text and "</html>" in r.text, route
        assert "<textarea" in r.text, f"{route} is not the chat page"
    about = client.get("/about")
    assert about.status_code == 200 and "<textarea" not in about.text


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


def test_markdown_renders_and_escapes():
    """The renderer's OUTPUT, not just that its calls resolve.

    `test_every_bare_call_in_the_page_js_resolves` passes on a renderer that emits
    nothing, and an escape bug here is an XSS in a page that displays model output.
    Runs the real JS under node; skips where node is absent (CI's cpu row has it, a
    bare pod may not) rather than asserting on the source text, which would prove
    only that the strings exist.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the renderer's output cannot be exercised")
    from tilerl.server import _MD_JS

    checks = [
        ("**b** and `c`", ["<strong>b</strong>", "<code>c</code>"]),
        ("# H\n\ntext", ["<h1>H</h1>", "<p>text</p>"]),
        ("- one\n- two", ["<ul>", "<li>one</li>"]),
        ("1. a\n2. b", ["<ol>", "<li>a</li>"]),
        ("> q", ["<blockquote>q</blockquote>"]),
        ("a <script>x</script> b", ["&lt;script&gt;"]),          # escape, never inject
        ("```py\nprint(1)\n```", ['<pre class="code"', 'data-lang="py"']),
        ("```\nunclosed", ['<pre class="code"']),                 # degrades, not swallows
        ("```\n**not bold**\n```", ["**not bold**"]),             # no inline rules in a fence
    ]
    harness = _MD_JS + "\nconst C = " + json.dumps(checks) + ";\n" + textwrap.dedent("""
        const bad = [];
        for (const [src, wants] of C) {
          const out = mdRender(src);
          for (const w of wants) if (!out.includes(w)) bad.push([src, w, out]);
        }
        if (mdRender("```\\n**x**\\n```").includes("<strong>")) bad.push(["fence", "isolation", ""]);
        console.log(JSON.stringify(bad));
    """)
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"the renderer threw: {r.stderr.strip()[:400]}"
    bad = json.loads(r.stdout.strip().splitlines()[-1])
    assert not bad, f"markdown output wrong: {bad}"
