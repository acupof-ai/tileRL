"""The served pages: the landing page string, and the built chat bundle.

The chat page used to be a 20 KB Python string, and the gates here were built
around that: slice the `<style>` block out of it, scan its script for bare calls,
run pieces of it under a stub DOM. It is now TypeScript under `web/`, built to
`src/tilerl/static/`, so three of those gates are gone because the compiler is
strictly stronger than they were -- `tsc --noEmit` runs on every build with
`strict`, `noUncheckedIndexedAccess` and `exactOptionalPropertyTypes`, and an
undefined call, a missing property or a wrong argument order is a build failure,
not a runtime one. What the compiler cannot see is what stayed: whether the page
the browser gets renders the frames the server actually sends.

That last one runs the REAL bundle -- the committed, minified artifact the server
serves -- against the REAL frames a real WebSocket connection produced. Both
halves shipped broken on 2026-09-04 with the other half correct, one in each
direction, so neither side may be a fixture written by hand.

# ponytail: node-only client gates skip where node is absent; a browser runner
# would also cover layout, which nothing here does
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tilerl.ui_assets import _LANDING

_STATIC = Path(__file__).resolve().parents[1] / "src" / "tilerl" / "static"


def _bundle() -> str:
    """The one JS artifact `index.html` loads.

    Read through the markup rather than globbed, so a stale asset left behind by an
    earlier build cannot be the thing under test while the server serves another.
    """
    html = (_STATIC / "index.html").read_text()
    srcs = re.findall(r'<script[^>]+src="\./assets/([\w.-]+\.js)"', html)
    assert len(srcs) == 1, f"index.html loads {srcs}, expected exactly one bundle"
    return (_STATIC / "assets" / srcs[0]).read_text()


def test_the_landing_page_js_parses():
    """`_LANDING`'s 261 bytes of inline JS, through a real parser.

    It is an ordinary triple-quoted string, so Python eats every backslash escape in
    it: a `\\n` inside a `//` comment once became a real newline, split the comment,
    and killed a whole script block. The chat page no longer has this hazard -- it is
    a file, not a string -- but the landing page still does.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the page JS cannot be parsed")
    js = _LANDING.split("<script>", 1)[1].split("</script>", 1)[0]
    r = subprocess.run([node, "--check", "-"], input=js, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, f"_LANDING's JS does not parse:\n{r.stderr.strip()[:600]}"


def test_the_landing_page_js_reads_only_ids_its_markup_defines():
    """An id the JS reads and the markup does not define is a null deref at load,
    and `_LANDING`'s two ids are the only thing between it and a header stuck on
    "connecting…"."""
    js = _LANDING.split("<script>", 1)[1].split("</script>", 1)[0]
    wanted = set(re.findall(r'getElementById\("([\w-]+)"\)', js))
    assert wanted, "the id reader matched nothing; the regex is stale"
    present = set(re.findall(r'id="([\w-]+)"', _LANDING))
    assert wanted <= present, f"_LANDING's JS reads ids the markup lacks: {wanted - present}"


def test_every_colour_token_is_defined_before_a_scheme_redefines_it():
    """A colour whose ONLY definition sits inside `prefers-color-scheme: dark`
    renders as nothing in light mode -- and the page still loads, still streams,
    and still passes every other gate here, so no existing test can see it.

    Checked in three directions on the SHIPPED html:
    every `var(--x)` resolves to a token the bare `:root` defines; the dark block
    only redefines tokens the bare block already has; and neither block leaves a
    token nothing reads (a dead token is a palette drifting out of step with the
    rules that were supposed to use it).
    """
    css = (_STATIC / "index.html").read_text()
    blocks = re.findall(r":root\s*\{([^}]*)\}", css)
    assert len(blocks) == 2, f"expected a bare :root and one scheme override, got {len(blocks)}"
    base = set(re.findall(r"(--[\w-]+)\s*:", blocks[0]))
    dark = set(re.findall(r"(--[\w-]+)\s*:", blocks[1]))
    used = set(re.findall(r"var\((--[\w-]+)\)", css))

    assert used <= base, (
        f"read but never defined in the bare :root: {sorted(used - base)} -- these render "
        f"as an empty value in light mode, which is the classic unreadable-artifact bug"
    )
    assert dark <= base, f"the dark block invents tokens the light one lacks: {sorted(dark - base)}"
    assert base <= used, f"defined but nothing reads them: {sorted(base - used)}"

    # And the direction ckl actually asked for: no pure white, no pure black.
    assert not re.search(r"#fff\b|#ffffff\b|#000\b|#000000\b", css, re.I), (
        "the palette is warm off-white on warm charcoal; a pure #fff or #000 slipped in"
    )


def test_the_bundle_and_the_markup_agree_on_every_id():
    """The bundle's `$` throws on a missing id rather than returning null, so one
    stale id is a blank page.

    Checked in both directions, and against the BUILT artifact rather than the sources:
    an id renamed in `index.html` without a rebuild leaves the served pair disagreeing
    while `web/src/` reads consistent. Minification renames the variable
    (`getElementById(t)`), so the ids are matched as the string literals they are
    passed in as.
    """
    html = (_STATIC / "index.html").read_text()
    bundle = _bundle()
    present = set(re.findall(r'id="([\w-]+)"', html))
    assert present, "index.html defines no ids; the markup is not what ships"
    read = {i for i in present if f'"{i}"' in bundle}
    assert read == present, (
        f"ids in the markup that the bundle never reads: {sorted(present - read)}. Either "
        f"the page grew dead markup or the bundle is stale -- rebuild with `npm run build`."
    )


def _strip_comments(ts: str) -> str:
    """Blank out comments and string literals.

    The sink gate scans for `innerHTML`, and `render.ts` explains in prose why it does
    not use one -- so the comment naming the hazard would fail the gate that exists
    because of it. Replaced with spaces rather than deleted, so nothing new becomes
    adjacent.
    """
    pattern = (
        r'"(?:[^"\\\n]|\\.)*"'      # double-quoted
        r"|'(?:[^'\\\n]|\\.)*'"     # single-quoted
        r"|`(?:[^`\\]|\\.)*`"       # template literal
        r"|//[^\n]*"                # line comment
        r"|/\*.*?\*/"               # block comment
    )
    return re.sub(pattern, lambda m: " " * len(m.group(0)), ts, flags=re.S)


def test_the_bundle_has_no_html_string_sink():
    """No markup can reach the DOM as text, so no reply can inject an attribute.

    This replaces the attribute-breakout gate the old renderer needed. That one fed 15
    crafted inputs through `mdRender` and parsed the output for attribute names outside
    an allow-list, because the renderer built an HTML STRING and the page assigned it to
    `innerHTML`: `[x](https://a"onmouseover="alert(1))` closed the href and the rest
    became a live handler. `render.ts` builds nodes with `createElement` and
    `createTextNode` instead, so text can only ever become text -- the class of bug is
    absent rather than defended against, and the allow-list has nothing left to guard.

    Checked over the built artifact and the sources, because either one could
    reintroduce it and only the artifact is what the browser runs.

    # ponytail: literal sinks only -- a computed `el["inner"+"HTML"]` is invisible here
    """
    sinks = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(")
    sources = sorted((_STATIC.parents[2] / "web" / "src").glob("*.ts"))
    assert sources, "no TypeScript sources found; this gate is looking in the wrong place"
    for label, text in [("the bundle", _bundle()),
                        *((p.name, _strip_comments(p.read_text())) for p in sources)]:
        found = [s for s in sinks if s in text]
        assert not found, (
            f"{label} writes markup as a string ({found}); model output reaches these "
            f"nodes, so an attribute breakout becomes reachable again"
        )


#: A DOM small enough to run the real bundle. Not a browser -- it answers the one
#: question no server-side assertion can: does what the server sent become what the
#: reader sees. `_html()` reconstructs the visible text of a subtree, since the
#: bundle builds nodes and never produces a string of its own.
_DOM_STUB = """
const mk = (tag) => ({
  tagName: tag.toUpperCase(), nodeValue: null, className: "", children: [],
  hidden: false, open: false, value: "", checked: false, disabled: false,
  classList: { _s: new Set(),
    add(...c){ c.forEach((x) => this._s.add(x)); },
    remove(...c){ c.forEach((x) => this._s.delete(x)); },
    contains(c){ return this._s.has(c); } },
  appendChild(c){ this.children.push(c); return c; },
  // Real ordering, not an append alias: `paint` inserts finished blocks BEFORE
  // the streaming tail, so a stub that ignored the ref node would hide a tail
  // that drifted out of last place.
  insertBefore(c, ref){ const i = this.children.indexOf(ref);
    const kids = c.tagName === "#FRAGMENT" ? c.children : [c];
    this.children.splice(i === -1 ? this.children.length : i, 0, ...kids); return c; },
  append(...c){ this.children.push(...c); },
  replaceChildren(...c){ this.children = c.flatMap((x) =>
    x.tagName === "#FRAGMENT" ? x.children : [x]); },
  addEventListener(ev, fn){ (this._h ||= {})[ev] = fn; },
  focus(){}, scrollIntoView(){},
  // Scroll geometry, so the page's "am I at the bottom" check has something to
  // read. A test sets scrollTop; scrollHeight/clientHeight are fixed, so
  // scrollTop === 900 is the bottom and anything less is scrolled back.
  scrollTop: 900, scrollHeight: 1000, clientHeight: 100,
});
const text = (v) => ({ tagName: "#TEXT", nodeValue: String(v), children: [] });
const IDS = {};
globalThis.document = {
  createElement: mk, createTextNode: text,
  createDocumentFragment: () => mk("#fragment"),
  getElementById: (i) => IDS[i] ?? null,
  addEventListener(){}, body: mk("body"),
};
globalThis.window = { location: { protocol: "http:", host: "x", href: "http://x/" },
  addEventListener(){} };
globalThis.location = globalThis.window.location;
// One socket, driven from the test: the bundle opens it, we replay the captured
// frames into onmessage, then close. No network, no timing.
globalThis.SENT = [];
globalThis.WebSocket = class {
  constructor(url){ globalThis.SOCK = this; this.url = url;
    queueMicrotask(() => this.onopen && this.onopen()); }
  send(d){ SENT.push(d); queueMicrotask(() => {
    for (const f of FRAMES) this.onmessage({ data: f });
    this.onclose && this.onclose();
  }); }
  close(){}
};
// The visible text of a subtree, tags included where they carry meaning.
const _html = (el) => el.tagName === "#TEXT" ? el.nodeValue
  : (el.tagName.startsWith("#") ? "" : `<${el.tagName.toLowerCase()}>`)
    + el.children.map(_html).join("")
    + (el.tagName.startsWith("#") ? "" : `</${el.tagName.toLowerCase()}>`);
const _text = (el) => el.tagName === "#TEXT" ? el.nodeValue : el.children.map(_text).join("");
"""


def test_the_websocket_route_never_emits_a_stop_sequence():
    """`_deltas` owns the stop cut for both transports, so the WS route gets it free.

    Two arms, one per layer, because the SSE arm alone does not cover this one:
    `_ScriptedEngine` honours `stop_texts` itself (`test_server.py`), so a route arm
    stays green with the *engine's* matching deleted — 48 measured all six route arms
    passing that way. This arm is red only when `_deltas`'s own streaming cut is
    reverted, which is the layer the WS route shares.

    The cut needs both halves, and each was verified to fail alone against
    `test_api_sdk.py::test_chat_stream_never_emits_the_stop_sequence`: with the whole
    cut deleted the stream leaks `"The answer is"`, and with only the holdback kept it
    leaks `"The answer "` — the trailing space of `" is"`, 48's own first error.
    """
    from test_server import _ByteTokenizer, _ScriptedEngine

    from tilerl.server import create_app

    tok = _ByteTokenizer()
    reply = "</think>\n\nThe answer is 4."
    app = create_app(_ScriptedEngine(tok, [reply]), tok)
    frames = []
    with TestClient(app) as c, c.websocket_connect("/ws/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 64, "enable_thinking": True, "stop": [" is"]})
        while True:
            f = ws.receive_json()
            frames.append(f)
            if f["t"] in ("done", "error"):
                break
    answer = "".join(f.get("content", "") for f in frames if f["t"] == "delta")
    assert answer == "The answer", (
        f"the stop sequence, or a prefix of it, reached the page: {answer!r}"
    )
    assert frames[-1]["t"] == "done" and frames[-1]["finish_reason"] == "stop", frames[-1]


def _ws_frames(replies: list[str], max_tokens: int, thinking: bool = True) -> list[str]:
    """The frames a real `/ws/chat` connection produces for `replies`.

    Over the app's own router and the app's own engine duck-type, so what the client
    gate replays is what the server emits rather than a shape someone typed here.
    """
    from test_server import _ByteTokenizer, _ScriptedEngine

    from tilerl.server import create_app

    tok = _ByteTokenizer()
    app = create_app(_ScriptedEngine(tok, replies), tok)
    frames: list[str] = []
    with TestClient(app) as c, c.websocket_connect("/ws/chat") as ws:
        ws.send_json({"messages": [{"role": "user", "content": "page"}],
                      "max_tokens": max_tokens, "enable_thinking": thinking})
        while True:
            f = ws.receive_json()
            frames.append(json.dumps(f))
            if f["t"] in ("done", "error"):
                break
    return frames


def test_the_websocket_route_streams_reasoning_then_the_answer():
    """The server half: the two phases arrive as their own fields, in order.

    `reasoning_content` and `content` are what the SSE route already sends (#159), so
    a reader of either transport learns one vocabulary. The page used to split on
    `</think>` itself, and once the server stripped the closer (#151) every reply
    landed whole in the reasoning fold over an empty bubble -- which is what "the V100
    returns no HTML" was.
    """
    frames = [json.loads(f) for f in
              _ws_frames(["planning\n</think>\n\n**hi**"], max_tokens=64)]
    kinds = [f["t"] for f in frames]
    assert kinds[-1] == "done" and "error" not in kinds, frames
    reasoning = "".join(f.get("reasoning_content", "") for f in frames)
    answer = "".join(f.get("content", "") for f in frames)
    # `split_think` keeps the newline before the closer as part of the reasoning; the
    # answer is the part after the blank line. Asserted verbatim, because a strip() here
    # would also pass on a split that dropped a whole line.
    assert reasoning == "planning\n", frames
    assert answer == "**hi**", frames
    # Ordering, not just presence: every reasoning frame precedes every content frame,
    # which is the property a page can fold on. Deleting the phase split in `_deltas`
    # interleaves them and fails here.
    last_r = max(i for i, f in enumerate(frames) if "reasoning_content" in f)
    first_c = min(i for i, f in enumerate(frames) if "content" in f)
    assert last_r < first_c, f"the two phases interleave: {frames}"
    assert frames[-1]["finish_reason"] == "stop", frames
    assert frames[-1]["usage"]["completion_tokens"] > 0, frames


def test_a_reply_cut_off_inside_the_block_says_length():
    """The state ckl hit: the budget spent inside `<think>`, so there is no answer.

    `finish_reason` is the only thing that separates it from a model that chose to say
    nothing, and the page writes a different notice for each. Forcing "stop" here
    makes the page call a truncation an empty reply.
    """
    reply = "still planning"
    from test_server import _ByteTokenizer

    frames = [json.loads(f) for f in
              _ws_frames([reply], max_tokens=len(_ByteTokenizer().encode(reply)))]
    assert frames[-1]["t"] == "done" and frames[-1]["finish_reason"] == "length", frames
    assert not any("content" in f for f in frames), f"an answer arrived: {frames}"
    assert "".join(f.get("reasoning_content", "") for f in frames) == reply, frames


def test_the_socket_neither_drops_nor_mislabels_the_fields_a_client_sends():
    """The WS route built its request from four hand-picked keys, so every other field
    the client sent vanished before pydantic ran -- and #201's `extra="allow"` could not
    see it either, because the extras never reached the constructor. Measured before the
    fix: temperature=0.5 and seed=7 arrived as None with no warning.

    Asserted at `SamplingParams`, not at the request object: what the defect actually
    broke is the value the ENGINE samples with, and `_ScriptedEngine` records the params
    of every submit.
    """
    import warnings

    from test_server import _ByteTokenizer, _ScriptedEngine

    from tilerl.server import create_app

    tok = _ByteTokenizer()
    engine = _ScriptedEngine(tok, ["</think>\n\nhi"])
    app = create_app(engine, tok)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with TestClient(app) as c, c.websocket_connect("/ws/chat") as ws:
            ws.send_json({"messages": [{"role": "user", "content": "page"}],
                          "max_tokens": 16, "enable_thinking": True,
                          "temperature": 0.5, "top_p": 0.9,
                          "a_field_we_do_not_declare": "xyz"})
            while ws.receive_json()["t"] not in ("done", "error"):
                pass
    texts = [str(w.message) for w in caught]

    # 1. the undeclared field is named -- #201's mechanism now reaches this route
    assert any("a_field_we_do_not_declare" in t for t in texts), (
        f"an undeclared field crossed the socket with no warning: {texts}")
    # 2. declared fields reach the ENGINE, which is what the key pick silently dropped
    params = engine.params[-1]
    assert params.temperature == 0.5, f"temperature never reached the engine: {params}"
    assert params.top_p == 0.9, f"top_p never reached the engine: {params}"
    # 3. `enable_thinking` must NOT warn: we honour it, so warning about it would be a
    #    fix that looks right and breaks the toggle.
    assert not any("enable_thinking" in t for t in texts), (
        f"warned about a field we honour: {texts}")
    # 4. and it is honoured by being MOVED, not dropped -- the prompt opens <think>.
    from tilerl.server import _ws_body
    moved = _ws_body({"enable_thinking": True})
    assert moved == {"chat_template_kwargs": {"enable_thinking": True}}, moved


def test_the_websocket_protocol_library_is_installed():
    """`TestClient.websocket_connect` fakes the transport in-process.

    So every assertion above passes with no WebSocket library installed at all, while
    the deployed server answers `/ws/chat` with **404** -- verified by holding a real
    `websockets` client constant and varying only the server's venv: without it,
    `InvalidStatus: HTTP 404`; with it, the socket connects. `serve_v100.sh` runs a
    plain venv, so this is the assertion that fails instead of the deploy.
    """
    import importlib.util

    assert importlib.util.find_spec("websockets") is not None, (
        "uvicorn serves /ws/chat only with a WebSocket protocol implementation "
        "installed; `uv sync --extra server` provides it"
    )


def _page_after(frames: list[str], budget: str = "",
                scroll_top: int | None = None, stop_after: bool = False) -> dict:
    """Run the shipped bundle over `frames`; return what landed in the DOM.

    ``budget`` is what the user typed in the budget box; "" is the shipped default
    (an empty box), which is what makes the ask omit ``max_tokens``.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; the page's reader cannot be executed")
    html = (_STATIC / "index.html").read_text()
    ids = sorted(set(re.findall(r'id="([\w-]+)"', html)))
    # The stub is populated from the page's own ids rather than a hand-kept list: a stub
    # missing one fails as a null deref, which reads like a page bug and is not one. The
    # two form defaults come from the markup for the same reason -- `checked` on the
    # thinking box is what makes the request ask for reasoning at all, and a stub that
    # hardcodes false would render a page that never has a fold to assert on.
    checked = set(re.findall(r'id="([\w-]+)"[^>]*\schecked', html))
    values = dict(re.findall(r'id="([\w-]+)"[^>]*\svalue="([^"]*)"', html))
    harness = (
        "const FRAMES = " + json.dumps(frames) + ";\n"
        + ("const SCROLLTOP = " + json.dumps(scroll_top) + ";\n" if scroll_top is not None else "")
        + "const STOP = " + ("true" if stop_after else "false") + ";\n"
        + _DOM_STUB
        + "".join(f'IDS["{i}"] = mk("div");\n' for i in ids)
        + "".join(f'IDS["{i}"].checked = true;\n' for i in sorted(checked))
        + "".join(f'IDS["{i}"].value = {v!r};\n'.replace("'", '"') for i, v in values.items())
        + f'IDS["budget"].value = {budget!r};\n'.replace("'", '"')
        + _bundle()
        + textwrap.dedent("""
        IDS.composer.value = "page";
        if (typeof SCROLLTOP === "number") IDS.log.scrollTop = SCROLLTOP;
        const done = IDS.send._h.click();
        // Mid-stream: the socket replays its frames on a microtask, so a click
        // scheduled here lands while the turn is still pending.
        if (STOP) IDS.stop._h.click();
        await done;
        await new Promise((r) => setTimeout(r, 0));
        const turn = IDS.log.children.at(-1);
        const fold = turn.children.find((c) => c.className === "reasoning");
        const answer = turn.children.find((c) => c.className === "answer");
        const note = turn.children.find((c) => c.className === "note");
        console.log(JSON.stringify({
          url: SOCK.url,
          scrollTop: IDS.log.scrollTop,
          stopHidden: IDS.stop.hidden,
          sent: JSON.parse(SENT[0]),
          reasoning: fold ? _text(fold.children[1]) : null,
          foldOpen: fold ? fold.open : null,
          answer: _html(answer),
          note: note.hidden ? null : _text(note),
          meter: _text(IDS.meter),
          pending: turn.classList.contains("pending"),
        }));
    """)
    )
    r = subprocess.run([node, "--input-type=module", "-e", harness],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"the page threw on the server's frames: {r.stderr.strip()[:800]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_an_empty_budget_box_sends_no_max_tokens_at_all():
    """Empty means "as much as fits", and OMITTING the field is the mechanism.

    The server reads absent/None as the context remainder
    (`min(max_total_tokens - prompt, 16*usable_blocks - prompt - width + 1)`), so a
    page that keeps sending its own 512 gets 512 and the change does nothing. A `0`
    or `""` is not the same thing either -- it serializes to `max_tokens: 0` and
    trips the `ge=1` validator.

    Two arms against the BUILT bundle: empty omits the key entirely, and a typed
    number still arrives verbatim. The second is the control -- an "omit always"
    regression passes the first arm alone.
    """
    empty = _page_after(_ws_frames(["ok"], max_tokens=16))["sent"]
    assert "max_tokens" not in empty, (
        f"the page sent max_tokens={empty.get('max_tokens')!r} with an empty box; the "
        f"server would use that instead of the context remainder"
    )
    assert empty["messages"], "the ask lost its messages while losing max_tokens"

    typed = _page_after(_ws_frames(["ok"], max_tokens=16), budget="128")["sent"]
    assert typed.get("max_tokens") == 128, (
        f"a typed budget must reach the server verbatim, got {typed.get('max_tokens')!r}"
    )


def test_the_renderer_builds_each_markdown_block_as_its_own_element():
    """Headings, lists, links and fences become real nodes, not styled text.

    One arm per construct, because a renderer that handles four of five looks
    identical to one that handles all five until the fifth appears in a reply.
    Asserted on tag names in the rendered subtree -- `_html` prints them -- rather
    than on the source text, which the old `.prose` path would also have passed.
    """
    reply = "\n".join([
        "## Result",
        "",
        "The answer is **391**, see [docs](https://example.com/x).",
        "",
        "- first",
        "- second",
        "",
        "1. one",
        "2. two",
        "",
        "```python",
        "print(17 * 23)",
        "```",
    ])
    got = _page_after(_ws_frames(["</think>\n" + reply], max_tokens=512))
    html = got["answer"]
    for tag in ("<h2>", "<ul>", "<li>", "<ol>", "<a>", "<pre>", "<code>", "<strong>"):
        assert tag in html, f"{tag} missing from the rendered answer: {html}"
    assert "## Result" not in html, f"the heading marker survived as text: {html}"
    assert "- first" not in html, f"the bullet marker survived as text: {html}"
    assert "print(17 * 23)" in html, f"the fenced body was lost: {html}"
    # The list markers are consumed, the text is not.
    assert "first" in html and "second" in html and "one" in html


def test_a_link_can_only_carry_a_scheme_we_allow():
    """`createElement` closes attribute breakout; it does NOT close `javascript:`.

    A node built with `a.href = "javascript:..."` is a live handler exactly as an
    injected attribute would be, so the string-vs-node argument that retired the
    old escaper does not cover this one. Anything but http/https/mailto/relative
    renders as plain text.

    Three arms: a hostile scheme is refused, an ordinary link still works (an
    "refuse everything" regression passes the first arm alone), and the refused
    link's TEXT is still shown rather than silently dropped.
    """
    bad = _page_after(_ws_frames(["</think>\nsee [click](javascript:alert(1)) here"],
                                 max_tokens=512))
    assert "<a>" not in bad["answer"], f"a javascript: href became a link: {bad['answer']}"
    assert "click" in bad["answer"], f"the refused link lost its text: {bad['answer']}"

    ok = _page_after(_ws_frames(["</think>\nsee [click](https://example.com) here"],
                                max_tokens=512))
    assert "<a>" in ok["answer"], f"an ordinary https link was refused: {ok['answer']}"


def test_a_fence_still_streaming_renders_as_code_not_as_a_paragraph():
    """An unterminated ``` is a code block whose body is what has arrived.

    Every frame repaints from the accumulated text, so mid-stream the last fence
    has no closer. Treating that as prose makes the block flip from paragraph to
    code when the closer lands -- the text reflows under the reader. The server
    here sends a reply that simply has no closing fence, which is the same input
    the renderer sees on every frame before the last.
    """
    got = _page_after(_ws_frames(["</think>\nintro\n\n```python\nprint(1)"], max_tokens=512))
    assert "<pre>" in got["answer"], f"an open fence rendered as prose: {got['answer']}"
    assert "print(1)" in got["answer"], got["answer"]
    assert "```" not in got["answer"], f"the fence marker leaked into the text: {got['answer']}"


def test_a_finished_block_is_not_rebuilt_by_a_later_frame():
    """Only the block still being written is re-parsed per frame.

    The whole answer used to be re-parsed and every node replaced on every frame:
    O(reply^2) over a stream, and it throws away the DOM under the reader's
    selection. `lastBlockStart` is the boundary -- everything before it is settled
    because the grammar's block breaks (a blank line, a closed fence) are already
    behind us.

    Driven through the real bundle: the reply has a finished paragraph, a closed
    fence and an open tail, so all three cases appear in one stream. The finished
    blocks must be siblings BEFORE the tail, which is the ordering `insertBefore`
    exists for.
    """
    reply = "</think>\nfirst para\n\n```py\nx = 1\n```\n\nstill typing"
    got = _page_after(_ws_frames([reply], max_tokens=512))
    html = got["answer"]
    # The settled paragraph and the closed fence both survived to the end.
    assert "first para" in html and "x = 1" in html and "still typing" in html, html
    assert "<pre>" in html, f"the closed fence did not become a code block: {html}"
    # The tail is last: everything settled precedes it.
    assert html.rindex("still typing") > html.rindex("x = 1"), (
        f"the streaming tail is not last; a finished block was inserted after it: {html}"
    )


def test_the_log_follows_the_stream_only_when_the_reader_is_at_the_bottom():
    """Scrolling someone away from the line they are reading is the bug here.

    Two arms, because a page that never scrolls passes the second alone and a page
    that always scrolls passes the first alone. The stub's geometry makes
    scrollTop 900 the bottom (scrollHeight 1000 - clientHeight 100).
    """
    frames = _ws_frames(["</think>\nsome reply text"], max_tokens=512)
    at_bottom = _page_after(frames, scroll_top=900)
    assert at_bottom["scrollTop"] == 1000, (
        f"a reader at the bottom stopped following the stream: {at_bottom['scrollTop']}"
    )
    scrolled_back = _page_after(frames, scroll_top=100)
    assert scrolled_back["scrollTop"] == 100, (
        f"the page yanked a reader who had scrolled back: {scrolled_back['scrollTop']}"
    )


def test_the_stop_button_is_shown_only_while_a_turn_is_in_flight():
    """It is `hidden` at rest and revealed on submit; the `finally` hides it again.

    Asserted after the stream settles rather than during it, which is the state a
    leak would show up in: a stop button still on screen with nothing to stop.
    """
    got = _page_after(_ws_frames(["</think>\nok"], max_tokens=512))
    assert got["stopHidden"] is True, "the stop button outlived the turn it belongs to"


def test_stopping_settles_the_turn_instead_of_raising():
    """A user stop is not a failure: the tokens already on screen are the reply.

    `ask` resolves on stop rather than rejecting, so the turn keeps its text and
    shows no error note. Driven by clicking stop mid-stream -- the stub's socket
    replays frames on a microtask, so the click lands while the turn is pending.
    """
    got = _page_after(_ws_frames(["</think>\npartial answer"], max_tokens=512), stop_after=True)
    assert got["note"] is None, f"a user stop rendered an error: {got['note']}"
    assert got["pending"] is False, "the turn stayed pending after a stop"


def test_the_page_renders_the_frames_this_server_sends():
    """The loop closed: the real bundle over a real connection's frames.

    Every other gate holds one side still -- the WS tests above assert on frames no
    page reads, and a hand-written fixture asserts on a page no server fed. So a change
    to the frame shape breaks the UI silently, which is how the page shipped broken
    twice on 2026-09-04 with the server correct throughout.
    """
    got = _page_after(_ws_frames(["planning\n</think>\n\n**hi** and `x`"], max_tokens=64))
    assert got["url"] == "ws://x/ws/chat", got
    assert got["sent"]["messages"][-1] == {"role": "user", "content": "page"}, got
    assert got["sent"]["enable_thinking"] is True, (
        f"the page did not read `checked` off the thinking box, so the reply it "
        f"rendered has no reasoning half to fold: {got}"
    )
    assert got["reasoning"] == "planning\n", got
    # The inner div is `.prose`, one per non-fenced run: markdown() emits block nodes,
    # so the answer bubble holds elements rather than a text blob.
    assert got["answer"] == "<div><div><p><strong>hi</strong> and <code>x</code></p></div></div>", got
    assert got["foldOpen"] is False, f"the fold opened over a finished answer: {got}"
    assert got["note"] is None, f"a healthy reply carries a notice: {got}"
    assert "completion_tokens" not in got["meter"] and got["meter"], got


def test_the_page_explains_a_reply_the_budget_cut_off():
    """An empty bubble is what ckl saw. The notice, and the reasoning left open,
    are the only things that say where the budget went."""
    reply = "still planning"
    from test_server import _ByteTokenizer

    got = _page_after(_ws_frames([reply], max_tokens=len(_ByteTokenizer().encode(reply))))
    assert got["reasoning"] == reply, got
    assert got["answer"] == "<div><div></div></div>", got
    assert got["foldOpen"] is True, f"the reasoning stayed folded over an empty reply: {got}"
    assert got["note"] and "budget" in got["note"], got


def test_a_typed_budget_spent_inside_the_block_names_the_number_the_user_typed():
    """The truncated notice quotes the TYPED cap when there is one, usage when not.

    The arm above drives the cap through the server fixture with the box empty, so
    it only ever exercises `cap ?? f.usage.completion_tokens` on the usage side.
    #194 made the box optional, which created a second path nothing covered: a
    user who types 12 must see 12, not the completion count that happens to equal
    it here by construction.

    So the number is made distinguishable on purpose -- the reply is longer than
    the typed cap, and the server is told a different, larger limit. If the page
    quoted usage instead of the typed value the notice would name that larger
    number and this fails.
    """
    from test_server import _ByteTokenizer

    reply = "still planning and planning"
    served = len(_ByteTokenizer().encode(reply))
    got = _page_after(_ws_frames([reply], max_tokens=served), budget="12")
    assert got["note"] is not None, f"a cut-off reply showed no notice: {got}"
    assert "12-token" in got["note"], (
        f"the notice must name the budget the user typed, got: {got['note']}"
    )
    assert str(served) not in got["note"], (
        f"the notice quoted the served completion count over the typed cap: {got['note']}"
    )
    assert got["foldOpen"] is True, "the reasoning stayed folded over an empty reply"


def _tiny_client():
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
    """`/` and `/chat` return the CHAT page; the landing page stays at /about.

    Asserting 200 + text/html + <title> cannot tell the two pages apart -- both satisfy
    all three -- so a route swap would be invisible. Key on the composer, which only the
    chat page has. `/chat` is its own route because `StaticFiles(html=True)` answers
    `/` with index.html and treats `/chat` as a missing file.
    """
    client = _tiny_client()
    for route in ("/", "/chat"):
        r = client.get(route)
        assert r.status_code == 200, route
        assert r.headers["content-type"].startswith("text/html"), route
        assert "<textarea" in r.text, f"{route} is not the chat page"
    about = client.get("/about")
    assert about.status_code == 200 and "<textarea" not in about.text
    # The mount at "/" is registered last and swallows anything after it, so check that
    # a route declared BEFORE it still answers rather than falling into the static dir.
    assert client.get("/health").status_code == 200, "the static mount shadowed the API"


def test_spacing_is_spent_from_a_scale():
    """23 hand-picked pixel values is what "the margins between components are wrong"
    looked like. Tokens need not cover every value -- optical padding is real -- but
    the common ones come from the scale."""
    css = (_STATIC / "index.html").read_text()
    css = css[css.index("<style>") : css.index("</style>")]
    tokens = set(re.findall(r"--s\d\b", css))
    assert len(tokens) >= 4, f"expected a spacing scale in :root, found {tokens}"
    uses = len(re.findall(r"var\(--s\d\)", css))
    assert uses >= 20, f"spacing scale declared but barely used ({uses} uses)"


def test_pending_survives_reduced_motion():
    """The caret is the only pending affordance and it is an animation, so
    `prefers-reduced-motion: reduce` used to leave the wait unsignalled."""
    css = (_STATIC / "index.html").read_text()
    i = css.index("@media (prefers-reduced-motion: reduce)")
    block = css[i : css.index("}", css.index("{", i)) + 1]
    assert "animation: none" in block and "content:" not in block, (
        "the reduced-motion block must drop the motion and keep the caret glyph"
    )


def test_no_style_rules_for_components_that_cannot_render():
    """#60 removed the tab strip and every event kind but `error`, and their CSS
    stayed. Dead spacing rules are the hardest kind to review -- they look like layout
    decisions for something you cannot find on the page."""
    html = (_STATIC / "index.html").read_text()
    css = html[html.index("<style>") : html.index("</style>")]
    bundle = _bundle()
    styled = set(re.findall(r"^\s*[\w.#:\-\[\]()= ]*?\.([a-z][\w-]*)", css, re.M))
    for cls in sorted(styled):
        assert f'"{cls}"' in bundle or f'class="{cls}"' in html or f"{cls} " in bundle, (
            f".{cls} is styled but nothing can render it"
        )


def test_gfm_tables_nested_lists_and_blockquotes_render():
    """The GFM constructs a reply actually contains, through the real bundle.

    ckl's report on the V100 page was "md 组件不全" -- the hand-rolled grammar covered
    headings, flat lists, fences, links and bold, so a table arrived as five lines of
    prose full of pipes, a nested list flattened to one level, and a blockquote kept its
    `>` as text. This is the case that was red before `marked`'s lexer replaced that
    grammar; it asserts the STRUCTURE (nesting, cell tags) rather than the text, because
    the text was always there -- it was the markup around it that was missing.

    One stream, four constructs, because they interact: a table's pipes must not be read
    as anything else, and the nested list has to survive the block boundary the table
    creates.
    """
    reply = (
        "</think>\n"
        "| op | ms |\n| --- | --- |\n| gemm | 1.2 |\n\n"
        "- outer\n  - inner\n\n"
        "> quoted\n\n"
        "- [x] done\n- [ ] todo\n\n"
        "~~gone~~ and `code`\n"
    )
    got = _page_after(_ws_frames([reply], max_tokens=512))
    a = got["answer"]
    for tag in ("<table>", "<thead>", "<th>", "<tbody>", "<td>", "<blockquote>"):
        assert tag in a, f"{tag} missing; GFM did not render: {a}"
    assert "<ul><li>" in a.replace(" ", ""), f"no list: {a}"
    # the nesting itself: an inner <ul> inside an <li>. The old renderer emitted two
    # flat items instead, which is why the text alone cannot be the assertion.
    assert re.search(r"<li>.*<ul>.*<li>.*inner", a, re.S), f"list did not nest: {a}"
    assert "<del>" in a, f"strikethrough missing: {a}"
    assert "<code>code</code>" in a, f"inline code missing: {a}"
    assert "gemm" in a and "1.2" in a, a
    # no marker leaked as text
    for marker in ("| ---", "~~", "> quoted"):
        assert marker not in a, f"{marker!r} reached the reader as text: {a}"
