"""The 27B's prompt format and sampling policy, shared by every route and the
trainer: ChatML with the checkpoint's XML tool calls, thinking switched in the
prompt, model-card sampling per thinking mode."""

from __future__ import annotations

import json
import re
import secrets
from typing import Any

# Qwen3.8-27B model card sampling per thinking mode (non-thinking also wants
# presence_penalty 1.5, which the engine does not have).
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}


def render_chat(messages: list[tuple[str, str]], thinking: bool | None = None) -> str:
    """ChatML from ``(role, text)`` pairs with the assistant turn left open.
    ``thinking`` follows the 27B template: True opens ``<think>``, False closes
    an empty one in the prompt, None leaves the bare turn (tiny/dev path)."""
    rendered = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in messages)
    tail = {None: "", True: "<think>\n", False: "<think>\n\n</think>\n\n"}[thinking]
    return f"{rendered}<|im_start|>assistant\n{tail}"


def render_prompt(messages: list[dict[str, Any]], system: Any = None,
                  tools: list[dict[str, Any]] | None = None,
                  thinking: bool | None = None, effort: str | None = None) -> str:
    """One prompt for Anthropic- or OpenAI-shaped ``messages`` (``role`` +
    string-or-block ``content``). The template puts the effort sentence and the
    tool block in the system turn, ahead of the caller's system text."""
    turns: list[tuple[str, str]] = []
    sys_text, tools_text = blocks_to_text(system), render_tools(tools, effort)
    if sys_text or tools_text:
        turns.append(("system", "\n\n".join(x for x in (tools_text, sys_text) if x)))
    for m in messages:
        turns.append((str(m.get("role", "user")), blocks_to_text(m.get("content"))))
    return render_chat(turns, thinking)


def sampling(tok: Any, thinking: bool | None, max_new_tokens: int, *,
             temperature: float | None = None, top_p: float | None = None,
             max_think_tokens: int | None = None, seed: int | None = None,
             logprobs: bool = False):
    """SamplingParams from the model card for this thinking mode; explicit
    ``temperature`` / ``top_p`` win. ``thinking=None`` (tiny/dev) samples at 1.0."""
    from .engine import SamplingParams

    kw = dict(SAMPLING[thinking]) if thinking is not None else {"temperature": 1.0}
    if temperature is not None:
        kw["temperature"] = temperature
    if top_p is not None:
        kw["top_p"] = top_p
    return SamplingParams(
        max_new_tokens=max_new_tokens, seed=secrets.randbits(31) if seed is None else seed,
        stop_token_ids=tuple(getattr(tok, "stop_token_ids", ())), logprobs=logprobs,
        # With thinking off the prompt closes the block itself, so no cap applies.
        max_think_tokens=max_think_tokens if thinking else None,
        end_think_ids=tuple(tok.encode("</think>\n\n")) if thinking else (), **kw)


def blocks_to_text(content: Any) -> str:
    """Flatten Anthropic content to the text a ChatML turn carries.

    tool_use renders as the checkpoint's own ``<tool_call>`` XML and tool_result
    as ``<tool_response>``, so a replayed transcript is byte-identical to what
    the model was trained on. Reasoning is stripped: the template re-inserts
    ``<think>`` only for turns after the last real user query, so feeding old
    reasoning back would be off-distribution.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return strip_think(content)
    out: list[str] = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        kind = b.get("type")
        if kind == "text":
            out.append(strip_think(b.get("text", "")))
        elif kind == "tool_use":
            out.append(render_tool_call(b.get("name") or "", b.get("input") or {}))
        elif kind == "tool_result":
            out.append(f"<tool_response>\n{blocks_to_text(b.get('content'))}\n</tool_response>")
        elif kind in ("image", "image_url", "document"):
            # Dropping these silently would send the model a turn that is
            # missing its subject; the 27B is text-only, so say so.
            raise ValueError(f"{kind} blocks are not supported by this model")
    return "\n".join(x for x in out if x)


#: Verbatim from the checkpoint's chat_template.jinja (read on the pod
#: 2026-09-02). Copied rather than paraphrased: this text is what the model was
#: trained to answer in, so a reworded version is a different distribution.
_TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is no "
    "function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n</IMPORTANT>"
)

#: The template's own wording per reasoning_effort; "medium" renders nothing.
_EFFORT_INSTRUCTIONS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, "
             "validate key assumptions, consider plausible alternatives, and prioritize "
             "correctness, consistency, and clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
           "directly to the conclusion without unnecessary elaboration.",
}

# A block cut off by max_tokens is still reasoning, not a reply.
_THINK_RE = re.compile(r"<think>.*?(?:</think>\s*|\Z)", re.S)


def split_think(text: str, opened: bool = False) -> tuple[str, str]:
    """(reasoning, reply) of assistant text; the reasoning block is dropped from the reply.

    ``opened``: the prompt already emitted ``<think>`` (the 27B template does when
    thinking is on), so the model's own text carries only the closer. Measured on
    the V100 endpoint: without this the reply began with the reasoning and a bare
    ``</think>``, and a client saw prose where it asked for HTML.
    """
    if opened and not text.lstrip().startswith("<think>"):
        text = "<think>" + text
    m = _THINK_RE.search(text)
    block = "" if m is None else m.group(0)[len("<think>"):]
    end = block.find("</think>")
    return (block if end < 0 else block[:end]), _THINK_RE.sub("", text)


def strip_think(text: str, opened: bool = False) -> str:
    return split_think(text, opened)[1]


def render_tool_call(name: str, args: dict[str, Any]) -> str:
    """One ``<tool_call>`` block, in the template's own shape.

    Non-string values are JSON, strings are raw -- exactly what the template's
    ``args_value | tojson`` branch does, so a replayed assistant turn matches
    what apply_chat_template would have produced.
    """
    lines = [f"<tool_call>\n<function={name}>"]
    for k, v in args.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        lines.append(f"<parameter={k}>\n{val}\n</parameter>")
    lines.append("</function>\n</tool_call>")
    return "\n".join(lines)


def render_tools(tools: list[dict[str, Any]] | None, effort: str | None = None) -> str:
    """The system turn's tool section, as the checkpoint's template builds it.

    Whole tool defs as JSON, one per line, inside ``<tools>`` -- not a trimmed
    summary. The 28 schemas Claude Code sends are most of the prompt, but the
    model was trained on the full defs and inventing a shorter form would be a
    format it has never seen.
    """
    head = _EFFORT_INSTRUCTIONS.get((effort or "").lower(), "")
    if not tools:
        return head
    body = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    for t in tools:
        body += "\n" + json.dumps(t, ensure_ascii=False)
    body += "\n</tools>" + _TOOL_INSTRUCTIONS
    return (head + "\n\n" + body) if head else body




def refuse_unsupported(*fields: str, **flagged: Any) -> None:
    """Raise on any field we accept but do not honour.

    A field taken and ignored is a format lie with a delayed cost: the client's
    NEXT request is built assuming the first one honoured it, so the failure
    surfaces a turn later and somewhere else. The same reason `blocks_to_text`
    raises on an image block rather than dropping it.

    Two forms, and the split is deliberate. Positional `fields` are already-decided
    refusals -- the caller tested the condition and passes the exact text to name,
    so `refuse_unsupported(*_unsatisfied_edits(cm))` can name one bad edit out of
    several. Keyword `flagged` is `name=<truthy?>`: it fires only when the client
    asked, and the NAME is what appears, never the value.

    An earlier version let a truthy string value replace the kwarg name, which made
    `previous_response_id="resp_1"` report "resp_1 is not supported" -- naming the
    value instead of the field, in three arms at once. Keeping the two forms apart
    is what makes that unrepresentable.

    Fields whose absence changes nothing observable are NOT passed by callers --
    `metadata` (echoed back) and `parallel_tool_calls` (we never emit parallel
    calls, so one call is a valid outcome either way).
    """
    named = list(fields) + [name for name, asked in flagged.items() if asked]
    if named:
        plural = "s are" if len(named) > 1 else " is"
        raise ValueError(
            f"{', '.join(sorted(named))}{plural} not supported by this server: the "
            f"request is refused rather than answered as if the field had been applied")
