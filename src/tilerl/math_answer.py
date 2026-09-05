"""Boxed-answer matching for MATH. `eval.answer_match` parses the last number and
MATH answers are not numbers -- `\\frac{1}{2}`, `2\\sqrt{3}`, `(3,\\frac{\\pi}{2})`
all score 0 against it, so a GRPO group on MATH would tie at the floor forever.

Exact match after normalisation, never a float compare: `\\frac{1}{3}` and 0.333
are different answers and a tolerance that merges them rewards a wrong one.
"""

from __future__ import annotations

import re

_TRAILING = re.compile(r"[.\s]+$")
_SPACE = re.compile(r"\s+")
# \left( \right] etc. are presentation, not value.
_DELIM = re.compile(r"\\(?:left|right|quad|qquad|[,!;:])")
_TEXT = re.compile(r"\\(?:text|mbox|mathrm)\s*\{([^{}]*)\}")


def extract_boxed(text: str | None) -> str | None:
    """The contents of the LAST ``\\boxed{...}``, brace-balanced.

    Brace-counted rather than regexed because the argument nests: ``\\boxed{\\frac{1}{2}}``
    ends at the fourth ``}``, and ``[^}]*`` stops at the first.
    """
    if not text:
        return None
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    i = start + len(r"\boxed{")
    depth = 1
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start + len(r"\boxed{") : i - 1] if depth == 0 else None


def normalize(ans: str | None) -> str | None:
    """Strip what LaTeX lets an answer vary by without changing its value."""
    if ans is None:
        return None
    s = _TEXT.sub(r"\1", ans)
    s = _DELIM.sub("", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\%", "").replace("%", "").replace("$", "")
    s = _SPACE.sub("", s)
    s = _TRAILING.sub("", s)
    # 0.50 and .5 are the same answer; \frac{1}{2} is deliberately NOT.
    if re.fullmatch(r"-?\d*\.?\d+", s):
        s = f"{float(s):g}"
    return s or None


def boxed_match(completion: str | None, answer: str) -> bool:
    got, want = normalize(extract_boxed(completion)), normalize(answer)
    return got is not None and want is not None and got == want


if __name__ == "__main__":
    assert extract_boxed(r"so \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"  # nested braces
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"  # last, not first
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{unclosed") is None
    assert boxed_match(r"answer is \boxed{2\sqrt{3}}", r"2\sqrt{3}")
    assert boxed_match(r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}")  # dfrac == frac
    assert boxed_match(r"\boxed{0.50}", ".5")  # numeric forms
    assert boxed_match(r"\boxed{\text{even}}", "even")
    assert not boxed_match(r"\boxed{\frac{1}{3}}", "0.333")  # NOT a float compare
    assert not boxed_match("the answer is 2", "2")  # unboxed does not count
    print("ok")
