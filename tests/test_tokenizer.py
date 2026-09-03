"""render_chat follows the 27B template: thinking is switched in the prompt.

Every expected string below was produced by rendering the checkpoint's own
chat_template.jinja with jinja2 on the pod, not derived by reading it. The template
raises on an unknown reasoning effort, so this does too.
"""

import pytest

from tilerl.tokenizer import render_chat

_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate "
    "key assumptions, consider plausible alternatives, and prioritize correctness, "
    "consistency, and clarity in the final answer."
)
_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
    "to the conclusion without unnecessary elaboration."
)


def test_render_chat_thinking_switch():
    turns = [("user", "hi")]
    base = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    # None: the tiny/dev path. A ByteTokenizer has no <think> token, so the turn stays
    # bare and no effort instructions are injected -- a state the checkpoint does not have.
    assert render_chat(turns) == base
    assert render_chat(turns, thinking=False) == base + "<think>\n\n</think>\n\n"
    # True is the checkpoint's default, and it injects the effort instructions.
    assert render_chat(turns, thinking=True) == (
        f"<|im_start|>system\n{_XHIGH}<|im_end|>\n" + base + "<think>\n"
    )


def test_reasoning_effort_matches_the_checkpoint_template():
    turns = [("user", "hi")]
    base = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"

    assert render_chat(turns, thinking=True, reasoning_effort="low") == (
        f"<|im_start|>system\n{_LOW}<|im_end|>\n" + base
    )
    # 'medium' is accepted and deliberately carries no instructions, so no system turn.
    assert render_chat(turns, thinking=True, reasoning_effort="medium") == base
    # An explicit effort with thinking off changes nothing: the template only reads it
    # when thinking is on.
    assert render_chat(turns, thinking=False, reasoning_effort="low") == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    with pytest.raises(ValueError, match="unexpected reasoning effort"):
        render_chat(turns, thinking=True, reasoning_effort="ultra")


def test_the_instructions_go_inside_the_callers_system_turn():
    """The template prepends, it does not add a second system turn."""
    turns = [("system", "You are X."), ("user", "hi")]
    assert render_chat(turns, thinking=True) == (
        f"<|im_start|>system\n{_XHIGH}\n\nYou are X.<|im_end|>\n"
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
