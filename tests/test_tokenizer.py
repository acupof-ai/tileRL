"""render_chat follows the 27B template: thinking is switched in the prompt."""

from tilerl.tokenizer import render_chat


def test_render_chat_thinking_switch():
    turns = [("user", "hi")]
    base = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert render_chat(turns) == base
    assert render_chat(turns, thinking=True) == base + "<think>\n"
    assert render_chat(turns, thinking=False) == base + "<think>\n\n</think>\n\n"
