"""The 27B's prompt format and sampling policy, shared by every route and the
trainer: ChatML with the checkpoint's XML tool calls, thinking switched in the
prompt, model-card sampling per thinking mode."""

from __future__ import annotations

import json
import re
import secrets
import warnings
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
        turns.append(_turn(m))
    return render_chat(turns, thinking)


def _turn(m: dict[str, Any]) -> tuple[str, str]:
    """One chat message as a ``(role, text)`` turn.

    ``role:"tool"`` re-enters as a USER turn carrying the same
    ``<tool_response>`` wrapper blocks_to_text gives an Anthropic
    tool_result block: the checkpoint template has no ``tool`` role, and the
    bare ``<|im_start|>tool`` would be off-distribution. An assistant turn
    carrying OpenAI ``tool_calls`` replays the calls' XML verbatim.
    """
    role = str(m.get("role", "user"))
    if role == "tool":
        return "user", f"<tool_response>\n{blocks_to_text(m.get('content'))}\n</tool_response>"
    text = blocks_to_text(m.get("content"))
    calls = m.get("tool_calls")
    if role == "assistant" and calls:
        parts = [text] if text else []
        parts += [render_tool_call_dict(tc) for tc in calls if isinstance(tc, dict)]
        text = "\n".join(parts)
    return role, text


def sampling(tok: Any, thinking: bool | None, max_new_tokens: int, *,
             temperature: float | None = None, top_p: float | None = None,
             max_think_tokens: int | None = None, seed: int | None = None,
             logprobs: bool = False, stop: Any = None):
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
        stop_texts=stop_texts(stop),
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
        if kind in ("text", "input_text", "output_text"):
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


def render_tool_call_dict(tc: dict[str, Any]) -> str:
    """One replayed tool call item, either OpenAI chat's nested
    ``{function:{name,arguments}}`` shape or a flat ``{name,arguments}``
    item (Responses ``function_call``). ``arguments`` is a JSON string on
    both wire shapes; an unparseable value renders empty args."""
    fn = tc.get("function") or tc
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {}
    return render_tool_call(fn.get("name") or "", args)


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




def stop_texts(stop: Any) -> tuple[str, ...]:
    """A route's ``stop`` / ``stop_sequences`` as engine ``stop_texts``.

    A bare string and a list are both documented shapes on the OpenAI routes.
    Empty entries are dropped rather than refused: "" matches at token 1, and a
    client that sends one meant "no stop", not "stop immediately".
    """
    items = [stop] if isinstance(stop, str) else list(stop or [])
    return tuple(s for s in items if isinstance(s, str) and s)


def cut_at_stop(text: str, stop: str | None) -> str:
    """The reply up to the stop sequence. The engine keeps the token that completed
    the match, so the text still carries it and the caller cuts at the match START
    -- OpenAI and Anthropic both exclude the sequence from the returned text."""
    return text if not stop else text.split(stop)[0]


#: Tool types that are the provider's to run, not ours. Declaring one means the
#: client expects the SERVER to perform the search or execution. Shared by the
#: chat and responses routes: accepting one silently renders a null-name tool
#: on chat while responses refuses the same request.
HOSTED_TOOL_TYPES = ("file_search", "web_search", "web_search_preview", "computer",
                     "computer_use_preview", "code_interpreter", "image_generation",
                     "local_shell", "mcp", "custom", "apply_patch", "shell")


def hosted_tool_fields(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The hosted tool types present in a request, as refusal kwargs."""
    kinds = {t.get("type") for t in tools or []} & set(HOSTED_TOOL_TYPES)
    return {f"tools[type={k}]": True for k in sorted(kinds)}


def unknown_fields(req: Any) -> dict[str, str] | None:
    """A request's undeclared fields, as name -> shape. Values are never recorded.

    Every request model declares ``extra="allow"`` so these survive parsing; without it
    pydantic drops them before any handler runs, and the recorded row is built from the
    parsed model, so a field we silently ignore is invisible in both. Not hypothetical:
    ``previous_response_id`` had to be DECLARED in order to be refused (see responses.py's
    own note), and it was found by reading the code, not by reading a log.

    Shape, not value: a body carries the user's prompt and may carry credentials, so a str
    becomes ``str[42]`` and a dict becomes ``dict{a,b}``. That is enough to tell a field we
    should honour from one we should refuse, and it cannot leak content.
    """
    extra = getattr(req, "model_extra", None)
    if not extra:
        return None
    shapes = {k: _shape(v) for k, v in sorted(extra.items())}
    # Warn too: the other two routes have no recorder, so this is their only signal.
    warnings.warn(
        f"{type(req).__name__}: ignoring undeclared request fields {shapes} -- declare one "
        "to honour it, or pass it to refuse_unsupported to reject it",
        stacklevel=2,
    )
    return shapes


def _shape(v: Any) -> str:
    if isinstance(v, bool) or v is None:  # bool before int: bool IS an int
        return repr(v)
    if isinstance(v, (int, float)):
        return f"{type(v).__name__}({v})"  # a number is its own shape, and the value matters
    if isinstance(v, str):
        return f"str[{len(v)}]"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, dict):
        # Keys say whether we should honour the field; values are content.
        return f"dict{{{','.join(sorted(map(str, v)))}}}"
    return type(v).__name__


#: reasoning_effort -> engine cap on <think> tokens. "none" (0) switches
#: thinking off in the prompt. The prompt TEXT vocabulary only knows
#: xhigh/medium/low; xhigh and max share the top cap with high -- the engine
#: budget has no higher step, and the template sentence is the route's job.
EFFORT_CAPS = {"none": 0, "minimal": 128, "low": 512, "medium": 2048,
               "high": 8192, "xhigh": 8192, "max": 8192}


def think_cap(effort: str | None) -> int | None:
    """The engine's <think> token cap for an effort value; None when absent."""
    if not effort:
        return None
    return EFFORT_CAPS.get(effort.lower())


def bad_effort(effort: str | None) -> bool:
    """An effort string outside every route's shared vocabulary."""
    return bool(effort) and effort.lower() not in EFFORT_CAPS


def effort_text(effort: str | None) -> str | None:
    """The prompt-text effort name: high/max are aliases of xhigh, which is
    the only instruction sentence the template has beyond low."""
    if not effort:
        return None
    return "xhigh" if effort.lower() in ("high", "max", "xhigh") else effort.lower()


def choice_name(choice: Any) -> str | None:
    """A ``tool_choice`` value's type name, either route's spelling.

    Both APIs accept a bare str or a ``{"type": ...}`` object.
    """
    if choice is None:
        return None
    return choice if isinstance(choice, str) else (choice or {}).get("type")


def tools_for_render(tools: list[dict[str, Any]] | None,
                     choice: Any) -> list[dict[str, Any]] | None:
    """The tools to render for a tool_choice: ``none`` forbids the call, so
    the tools block must not reach the prompt -- accepting ``none`` while
    still rendering the tools made the field a lie."""
    return None if choice_name(choice) == "none" else tools


def unsupported_choice(choice: Any) -> bool | None:
    """`tool_choice` beyond auto/none, for either route's spelling.

    Both APIs accept a str or a `{"type": ...}`; Anthropic's `any` and `tool` and
    OpenAI's `required`/`{type: function}` all force a call. We render tools into the
    prompt and cannot force or forbid one, so anything stronger than a hint is
    unimplementable and is refused rather than dropped.
    """
    name = choice_name(choice)
    return None if name is None else name not in ("auto", "none")


def refuse_unsupported(*fields: str, **flagged: Any) -> None:
    """Raise on a field we accept but do not honour: the client's NEXT request
    assumes the first one applied it, so the lie surfaces a turn later.

    Positional args are the exact text to name (one bad item out of several);
    keyword args are `name=<truthy?>` and name themselves, never the value --
    apart, so a value like "resp_1" cannot end up as the field name.
    """
    named = list(fields) + [name for name, asked in flagged.items() if asked]
    if named:
        plural = "s are" if len(named) > 1 else " is"
        raise ValueError(
            f"{', '.join(sorted(named))}{plural} not supported by this server: the "
            f"request is refused rather than answered as if the field had been applied")


#: Idle gap between take/peek polls of a blocked row. One value for the
#: non-stream waiter and the SSE peek loop, not two copies of the same 0.02.
POLL_INTERVAL_S = 0.02


def await_completion(engine: Any, request_id: int, timeout_s: float,
                     poll_s: float = POLL_INTERVAL_S) -> list[int]:
    """Block until ``engine.take`` returns the row, or raise TimeoutError.

    The single wait body every non-stream route runs (inside asyncio.to_thread
    via server.await_or_cancel, which owns disconnect polling and cancellation).
    take() pops only this request — poll() would steal another row's completion.
    ``timeout_s <= 0`` removes the deadline (long-context server mode); the
    ASGI disconnect watcher still interrupts the wait, so a hung client is not
    waited on forever.
    """
    import time

    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while deadline is None or time.monotonic() < deadline:
        out = engine.take(request_id)
        if out is not None:
            return out
        time.sleep(poll_s)
    raise TimeoutError(f"request {request_id} did not finish within {timeout_s}s")


def flatten_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """OpenAI/Responses tool dicts as the flat {name, description, input_schema}
    vocabulary the template renders and messages._parse_tool_calls reads, so a
    call parses identically whichever API declared it.

    Accepts OpenAI's ``{type, function: {name, description, parameters}}`` and
    Responses' top-level ``{name, description, parameters|input_schema}``.
    """
    if not tools:
        return None
    out = []
    for t in tools:
        fn = t.get("function") or t
        out.append({"name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or fn.get("input_schema") or {}})
    return out


def thinking_enabled(tokenizer: Any, explicit: bool | None, effort_none: bool) -> bool | None:
    """Common tail of the three routes' thinking adapters.

    An explicit caller override wins; an effort of "none" switches thinking off;
    otherwise the template's own default, which needs a tokenizer that HAS the
    ``<think>`` tag (ByteTokenizer makes one token per byte, and its bare turn is
    the dev path with no such tag). The per-API input adapters (raw dict /
    MessagesRequest / ResponsesRequest) stay in the routes and feed this.
    """
    if explicit is not None:
        return bool(explicit)
    if effort_none:
        return False
    return len(tokenizer.encode("<think>")) == 1 or None
