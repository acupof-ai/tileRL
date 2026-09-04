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

from tilerl.server import _CHAT_UI, _LANDING

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
    """The FIRST inline script in `html`.

    `rsplit("</script>")` took the LAST closing tag, so a page with two script blocks
    returned everything between the first open and the final close -- measured on
    _LANDING + _CHAT_UI, 27299 characters of markup captured as JavaScript, which the
    resolver would then scan for bare calls. Correct today only because every caller
    hands this one page; `split` makes it correct regardless.
    """
    assert "<script>" in html, "the page has no inline script"
    return html.split("<script>", 1)[1].split("</script>", 1)[0]


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


#: A DOM small enough to run the page's own send path. Not a browser -- it answers one
#: question the other gates cannot: does clicking send actually reach fetch, and does
#: the fold's summary keep the structure that makes it clickable.
_DOM_STUB = """
const mk = (tag) => ({
  tagName: tag.toUpperCase(), className: "", textContent: "", innerHTML: "", children: [],
  style: {}, dataset: {}, open: false, value: "", disabled: false, hidden: false,
  scrollTop: 0, scrollHeight: 0,
  classList: { add(){}, remove(){}, toggle(){} },
  appendChild(c){ this.children.push(c); c.parentNode = this; return c; },
  insertBefore(c){ this.children.push(c); c.parentNode = this; return c; },
  querySelector(sel){
    const hit = (e) => ("." + e.className) === sel ? e : e.children.map(hit).find(Boolean);
    return hit(this) || mk("span");
  },
  querySelectorAll(){ return []; },
  addEventListener(ev, fn){ (this._h ||= {})[ev] = fn; },
  setAttribute(){}, getAttribute(){}, focus(){}, remove(){}, scrollIntoView(){},
});
globalThis.document = { createElement: mk, getElementById: (i) => IDS[i] ?? null,
  querySelector: () => mk("div"), querySelectorAll: () => [],
  addEventListener(){}, body: mk("body") };
globalThis.window = { addEventListener(){}, location: { origin: "http://x" } };
globalThis.performance = { now: () => 0 };
const CALLS = [];
globalThis.fetch = async (u, o) => { CALLS.push({ u, body: o && o.body });
  return { ok: true, status: 200, body: { getReader: () => ({ read: async () => ({done: true}) }) } }; };
globalThis.AbortController = class { constructor(){ this.signal = {}; } abort(){} };
globalThis.TextDecoder = class { decode(){ return ""; } };
"""


@pytest.mark.parametrize("name,page", [("_CHAT_UI", _CHAT_UI), ("_LANDING", _LANDING)])
def test_the_page_js_parses(name, page):
    """Each page's whole script block, through a real parser.

    Both pages are ordinary triple-quoted strings, so Python eats every backslash
    escape in them. A `\\n` inside a `//` comment became a real newline and split the
    comment in two, leaving `", so` as a statement: SyntaxError, the entire script
    block dead, and a page that renders but cannot send. The bare-call resolver saw
    nothing wrong -- it scans text and does not parse -- and every other gate here
    passed. The served page was broken for a whole deploy.

    Parametrized over both pages because `_LANDING`'s 261 bytes of JS carry the same
    hazard and had no gate of any kind. This is also why _MD_JS is `r\"\"\"`; the same
    escape class had already cost a working regex earlier the same day.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the page JS cannot be parsed")
    js = _script(page)
    r = subprocess.run([node, "--check", "-"], input=js, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, f"{name}'s JS does not parse:\n{r.stderr.strip()[:600]}"
    # A stray escape shows up as a line that is the tail of a broken string literal.
    # Cheap, runs without node, and names the cause rather than a parse offset.
    for n, line in enumerate(js.splitlines(), 1):
        assert not line.lstrip().startswith('", '), (
            f"{name} line {n} is {line.strip()!r}: a backslash escape was eaten by "
            f"Python and split a comment or string. These pages are not raw strings."
        )


@pytest.mark.parametrize("name,page,reader", [
    ("_CHAT_UI", _CHAT_UI, r'\$\("([\w-]+)"\)'),
    ("_LANDING", _LANDING, r'getElementById\("([\w-]+)"\)'),
])
def test_every_element_the_js_reads_exists_in_the_markup(name, page, reader):
    """An id the JS reads and the markup does not define is a null deref at load.

    On the chat page that kills the whole script -- the same shape as the parse break,
    caught only at runtime. `_LANDING` had no gate at all, and its two ids are the
    only thing between it and a page whose header never leaves "connecting…".
    """
    wanted = set(re.findall(reader, _script(page)))
    assert wanted, f"{name}: the id reader matched nothing; the regex is stale"
    present = set(re.findall(r'id="([\w-]+)"', page))
    assert wanted <= present, f"{name}'s JS reads ids the markup lacks: {wanted - present}"


def test_sending_reaches_fetch_and_the_fold_stays_clickable():
    """The page's own send path, executed.

    Two failures got past every text-level gate here and reached the user: the script
    block did not parse at all, and `display: flex` on the <summary> cost it the
    disclosure behaviour, so the fold rendered and would not open. Both are only
    visible if the code runs, so run it -- against a stub DOM, no browser.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the send path cannot be executed")
    # Populate the stub from the page's own ids rather than a hand-kept list -- a stub
    # missing one id fails as a null deref, which reads like a page bug and is not one.
    # That they all exist in the markup is its own test above.
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', _script(_CHAT_UI)))
    ids = "const IDS = {};\n" + "".join(f'IDS["{i}"] = mk("div");\n' for i in sorted(wanted))
    harness = _DOM_STUB + ids + _script(_CHAT_UI) + textwrap.dedent("""
        const out = {};
        const bubble = addMsg("user", "hello");
        out.addMsg = bubble ? bubble.tagName : null;
        out.feed = IDS.feed.children.length;
        await sendChat("hi");
        // The page also probes /v1/models on load, which carries no body -- pick the
        // completion POST rather than assuming it is the first call.
        const post = CALLS.filter((c) => c.body);
        out.posts = post.length;
        out.url = post.length ? post[0].u : null;
        out.thinking = post.length ? post[0].body.includes("enable_thinking") : false;
        // The fold: <details><summary><span class="row">..., because a flex summary
        // is not clickable.
        const body = addThinking(addMsg("assistant", ""));
        const det = body.parentNode;
        out.summary = det.children[0].tagName;
        out.row = det.children[0].children[0].className;
        out.chev = det.children[0].children[0].children[0].className;
        console.log(JSON.stringify(out));
    """)
    r = subprocess.run([node, "--input-type=module", "-e", harness],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"the page threw when run: {r.stderr.strip()[:600]}"
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["addMsg"] == "DIV" and got["feed"] >= 1, f"addMsg built nothing: {got}"
    assert got["posts"] == 1, f"send did not reach fetch: {got}"
    assert got["url"] == "/v1/chat/completions", f"send posted to {got['url']!r}"
    assert got["thinking"], f"the request does not ask for thinking: {got}"
    assert got["summary"] == "SUMMARY", f"the fold is not a summary: {got}"
    assert got["row"] == "row" and got["chev"] == "chev", (
        f"the summary lays itself out instead of an inner row, which is what made the "
        f"fold unclickable: {got}"
    )


def test_the_slicers_take_one_block_from_a_two_block_page():
    """`server.py` holds two pages, so both slicers can over-capture.

    `_script` used rsplit, which on a two-block page returns everything from the first
    open tag to the LAST close tag: 27299 characters of markup handed to the resolver as
    JavaScript. `_css` has the same shape and its docstring warns about it. Neither
    hazard is reachable today -- every caller passes `_CHAT_UI` -- so this is the gate
    that fails if a page merge makes it reachable.
    """
    page = _LANDING + _CHAT_UI
    assert "</script>" not in _script(page), "_script spans past its own block"
    assert "</style>" not in _CHAT_UI[_CHAT_UI.index("<style>") : _CHAT_UI.index("</style>")]
    # And the real page still yields the script the other tests assert on.
    assert "function mdRender" in _script(_CHAT_UI)


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


def _tiny_client():
    """A TestClient over the real app on the tiny model.

    Two tests need it; building the engine twice doubles the slowest part of this file.
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
    return TestClient(create_app(engine, get_tokenizer(None), model_name="tiny"))


def test_the_index_route_serves_the_page():
    """`/` returns the CHAT page, and the landing page is still reachable at /about.

    Asserting 200 + text/html + <title> cannot tell the two pages apart -- both satisfy
    all three -- so the route swap would have been invisible. Key on the composer, which
    only the chat page has.
    """
    client = _tiny_client()
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


def test_no_markdown_input_can_add_an_attribute_to_the_output():
    """Attribute breakout: the renderer's output goes to innerHTML.

    `mdEscape` escaped `&`, `<`, `>` and no quotes, while two rules interpolate its
    result into an HTML ATTRIBUTE -- `href="..."` in the link rule and `data-lang="..."`
    on a fence. So a quote in a URL or a fence language closed the attribute and the
    rest of the token became markup. Executed against the real renderer, both produced
    live event handlers.

    Asserts on the attribute NAMES in the output, not on substrings. A substring search
    cannot tell an attribute from text -- `&quot;onmouseover=&quot;` contains the handler
    name and is inert -- and that difference is exactly what the fix creates. It is also
    what caught the half the first fix missed: escaping only `"` left the single-quote
    variant parsing as a real `onmouseover` attribute.

    The allow-list is closed rather than a deny-list of `on*`: `style`, `srcdoc` and
    `formaction` are all reachable without an event handler name.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the renderer's output cannot be exercised")
    from tilerl.server import _MD_JS

    #: Every attribute the renderer is allowed to emit, over every input.
    allowed = {"href", "target", "rel", "class", "data-lang"}
    attacks = [
        '[x](https://a"onmouseover="alert(1))',
        "[x](https://a'onmouseover='b)",
        '```js"onload="alert(1)\ncode\n```',
        "```js'onload='alert(1)\ncode\n```",
        '[x](https://a"style="width:99vw)',           # not an on* name
        '[x](javascript:alert(1))',                    # scheme, not breakout
        '# h"onclick="x',
        '- item"onclick="x',
        '> quote"onclick="x',
        '**b"onclick="x**',
        # A raw tag, so this gate covers the `<` escape too and not only the quotes.
        # Without it, dropping the `<` escape read MISSED under mutation while the
        # quote mutations were caught.
        '<img src=x onerror=alert(1)>',
        '<a href="javascript:alert(1)">x</a>',
        # Benign inputs must keep working: a regression here is a broken page, and a
        # gate that only feeds attacks cannot see it.
        '[ok](https://e.com/a?b=1&c=2)',
        '```py\nprint(1)\n```',
        'it\'s a "test"',
    ]
    harness = _MD_JS + "\nconst C = " + json.dumps(attacks) + ";\n" + textwrap.dedent("""
        const out = [];
        for (const s of C) {
          const html = mdRender(s);
          const attrs = new Set();
          // Tag interiors only, so attribute-looking text in the body is not counted.
          for (const tag of html.match(/<[a-z][^>]*>/g) || [])
            for (const m of tag.matchAll(/[\\s"']([a-zA-Z-]+)\\s*=/g)) attrs.add(m[1]);
          out.push([s, [...attrs], html]);
        }
        console.log(JSON.stringify(out));
    """)
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"the renderer threw: {r.stderr.strip()[:400]}"
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    for src, attrs, html in rows:
        extra = sorted(set(attrs) - allowed)
        assert not extra, (
            f"input {src!r} put {extra} into the markup, which innerHTML will honour:\n  {html}"
        )
    # The benign rows must still render, or an over-eager escape passes this vacuously.
    by_src = {src: html for src, _, html in rows}
    assert 'href="https://e.com/a?b=1&amp;c=2"' in by_src['[ok](https://e.com/a?b=1&c=2)']
    assert 'data-lang="py"' in by_src['```py\nprint(1)\n```']


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
