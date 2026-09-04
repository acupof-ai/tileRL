"""The served chat page: 480 lines of CSS and a streaming state machine, gated.

Nothing rendered ``/`` before this file. `_CHAT_UI` is a single string constant,
so every defect in it is invisible to the test suite -- two sessions in a row
reviewed a version of this page that had already been replaced, because there
was no gate to fail when it moved.

These are structural assertions, not a snapshot: a snapshot of a 20 KB string
fails on every edit and teaches nothing. Each test names one property the page
must keep and fails only when that property breaks.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from tilerl.server import _CHAT_UI


def _css() -> str:
    """The chat page's <style> block alone.

    `server.py` holds two pages; slicing from the file's first `<style>` picks
    up the landing page instead, whose spacing is its own concern. Slice from
    `_CHAT_UI` itself so this can only ever describe the chat page.
    """
    head = _CHAT_UI.index("<style>")
    return _CHAT_UI[head : _CHAT_UI.index("</style>", head)]


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
