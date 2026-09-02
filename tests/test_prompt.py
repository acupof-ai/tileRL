"""The prompt is the 27B's template byte for byte; both routes render through
one function, so the golden strings below (captured from the routes before they
were unified) are the gate against a silent reordering."""

from tilerl.prompt import render_chat, render_prompt, sampling

ANTHROPIC = "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"name\": \"Bash\", \"description\": \"Run\", \"input_schema\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}, \"timeout\": {\"type\": \"integer\"}}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou are tileRL.<|im_end|>\n<|im_start|>user\nls it<|im_end|>\n<|im_start|>assistant\nok\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\na\nb\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
OPENAI = "<|im_start|>system\nYou are tileRL.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n<|im_start|>user\nagain<|im_end|>\n<|im_start|>assistant\n"


def test_render_chat_thinking_switch():
    turns = [("user", "hi")]
    base = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert render_chat(turns) == base
    assert render_chat(turns, thinking=True) == base + "<think>\n"
    assert render_chat(turns, thinking=False) == base + "<think>\n\n</think>\n\n"


def test_anthropic_route_prompt_is_unchanged():
    tools = [{"name": "Bash", "description": "Run", "input_schema": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}}}}]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "ls it"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"},
                                          {"type": "tool_use", "id": "t1", "name": "Bash",
                                           "input": {"command": "ls", "timeout": 30}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a\nb"}]}]
    out = render_prompt(messages, [{"type": "text", "text": "You are tileRL."}], tools, True, "low")
    assert out == ANTHROPIC


def test_openai_route_prompt_is_unchanged():
    messages = [{"role": "system", "content": "You are tileRL."},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "yo"}, {"role": "user", "content": "again"}]
    assert render_prompt(messages) == OPENAI


def test_openai_route_now_follows_the_template():
    # Two deliberate changes vs the old OpenAI route: multi-part content joins
    # with a newline (was concatenated) and stale <think> text is stripped.
    out = render_prompt([{"role": "user", "content": [{"type": "text", "text": "a"},
                                                      {"type": "text", "text": "b"}]},
                         {"role": "assistant", "content": "<think>x</think>ans"},
                         {"role": "user", "content": "next"}], thinking=False)
    assert out == ("<|im_start|>user\na\nb<|im_end|>\n<|im_start|>assistant\nans<|im_end|>\n"
                   "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_sampling_explicit_values_beat_the_card():
    class Tok:
        stop_token_ids = (7,)

        def encode(self, s):
            return [1, 2]

    p = sampling(Tok(), True, 8)
    assert (p.temperature, p.top_p, p.top_k, p.end_think_ids) == (1.0, 0.95, 20, (1, 2))
    p = sampling(Tok(), True, 8, temperature=0.0, top_p=0.5)
    assert (p.temperature, p.top_p, p.top_k) == (0.0, 0.5, 20)
    p = sampling(Tok(), None, 8)
    assert (p.temperature, p.top_k, p.end_think_ids, p.max_think_tokens) == (1.0, 0, (), None)
