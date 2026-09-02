"""Thinking effort on the real checkpoint: same prompt at several budgets.

  python scripts/think_effort.py /data00/Qwen3.8-27B-NVFP4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tilerl.config import qwen38_27b  # noqa: E402
from tilerl.engine import SamplingParams, build_engine  # noqa: E402
from tilerl.model import load_hf  # noqa: E402
from tilerl_kernels.backend import get_backend  # noqa: E402
from tilerl.server import get_tokenizer  # noqa: E402

src = sys.argv[1]
backend = get_backend()
model = load_hf(qwen38_27b(), src, fuse_projections=True)
engine = build_engine(model.cfg, model, backend, num_blocks=1024, num_slots=8, max_batch=4,
                      max_total_tokens=8192)
tok = get_tokenizer(src)
end = tuple(tok.encode("</think>\n\n"))
prompt = tok.encode("<|im_start|>user\nWhat is 17 * 23?<|im_end|>\n<|im_start|>assistant\n")
print("end_think_ids:", end, "->", repr(tok.decode(list(end))))
for label, budget in (("unset", None), ("none", 0), ("minimal", 32)):
    sp = SamplingParams(temperature=0.0, max_new_tokens=96, seed=0,
                        max_think_tokens=budget, end_think_ids=end)
    wid = engine.submit(prompt, sp)
    out = None
    while out is None:
        engine.step()
        out = engine.poll().get(wid)
    print(f"\n--- reasoning_effort={label} (budget={budget})\n{tok.decode(out)!r}")
