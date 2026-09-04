"""Every bare call in the chat page's JS must resolve to something defined.

`addThinking` shipped called-but-never-defined and no gate could see it: the whole suite
asserts on SSE bytes, and `_CHAT_UI` is 21 KB of JavaScript that Python only measures the
length of. `node --check` does NOT close this -- verified: it reports SYNTAX_OK on
`function a(){ return undefinedFn(1); }`, because an undefined call is a runtime error.

So this check needs no JS runtime and therefore always runs in CI. It is deliberately
syntactic: collect identifiers called WITHOUT a leading dot (a bare call, not a method),
subtract what the script defines and the globals a browser provides, and fail on the rest.

# ponytail: undefined calls only, stub-DOM execution when the page grows
"""

import re

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


def _script(html: str) -> str:
    assert "<script>" in html, "the page has no inline script"
    return html.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def _code_only(js: str) -> str:
    """Blank out string literals, template literals and comments.

    Without this, UI copy triggers false positives: `"read/write files  (Enter to send)"`
    reads as a call to `files`. Measured on HEAD~1 -- it reported `files` and `tilerl`
    alongside the real `addThinking`. Replaced with spaces, not deleted, so nothing new
    becomes adjacent.
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
    # UI copy must not read as code. Measured: before literals were stripped, HEAD~1
    # reported `files` and `tilerl` from a placeholder string alongside the real bug.
    assert _unresolved('let p = "read/write files  (Enter to send)";') == set()
    assert _unresolved("// call ghost(1) in a comment\nlet a = 1;") == set()


if __name__ == "__main__":
    test_every_bare_call_in_the_page_js_resolves()
    test_the_check_catches_an_undefined_call()
    print("ok")
