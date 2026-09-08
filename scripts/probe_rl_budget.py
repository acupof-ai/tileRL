"""Where do 91.8 of 95.2 GiB go before the first optimizer step?

The ISO-RL arm OOMed three times. The last one asked for 4.74 GiB (the f32 embedding copy in
`autograd.step_one`) with 3.38 GiB free. But 25's 27B arm on the same card type peaks at 25.99
GiB and leaves 67.40 free, so the interesting number is not the 4.74 -- it is the 64 GiB
difference nobody has explained. A lever that frees 0.38 or 4.74 GiB is subtraction inside a
budget we do not understand.

So: the inventory, at each stage of what the arm actually builds, by dtype and by device. No
training step, no optimizer -- this is a read, not a run.
"""
import os
from collections import defaultdict

import torch

os.environ.setdefault("TILERL_TARGET", "cuda")

from tilerl_kernels.backend import get_backend  # noqa: E402

from tilerl.cli import _build_model  # noqa: E402
from tilerl.engine import build_engine  # noqa: E402
from tilerl.kv_cache import NoPrefixStore  # noqa: E402

GiB = 2**30


def report(tag):
    a = torch.cuda.memory_allocated() / GiB
    r = torch.cuda.memory_reserved() / GiB
    free, total = (x / GiB for x in torch.cuda.mem_get_info())
    print(f"{tag:40} allocated {a:7.2f}  reserved {r:7.2f}  free {free:6.2f} of {total:.2f}")


def inventory(params, tag):
    """By device and dtype, since an f32 master beside the served bytes is the suspicion."""
    by = defaultdict(lambda: [0, 0])
    for v in params.values():
        k = (str(v.device).split(":")[0], str(v.dtype))
        by[k][0] += 1
        by[k][1] += v.numel() * v.element_size()
    print(f"\n{tag}: {len(params)} tensors")
    for (dev, dt), (n, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {dev:6} {dt:18} x{n:5}  {b/GiB:7.2f} GiB")
    on_card = sum(b for (dev, _), (_, b) in by.items() if dev == "cuda")
    print(f"  card total from params: {on_card/GiB:.2f} GiB")
    return on_card


report("start")
cfg, model = _build_model("qwen38-27b", seed=0, keep_master=True)
report("after _build_model(keep_master=True)")
inventory(model.params, "model.params after build")

backend = get_backend()
report("after get_backend")

engine = build_engine(cfg, model, backend, num_blocks=256, num_slots=8,
                      decode_graph=False, prefix_store=NoPrefixStore())
report("after build_engine")
on_card = inventory(model.params, "model.params after build_engine")

# step_one does p32 = p.to(float32) per parameter, so the peak ADD is the largest single
# parameter, not the sum.
biggest = max(model.params.values(), key=lambda v: v.numel())
name = next(k for k, v in model.params.items() if v is biggest)
print(f"\nlargest param: {name} {tuple(biggest.shape)} {biggest.dtype}")
print(f"  its f32 copy: {4*biggest.numel()/GiB:.2f} GiB  <- what step_one asks for")
free = torch.cuda.mem_get_info()[0] / GiB
print(f"  free right now: {free:.2f} GiB  -> fits: {4*biggest.numel()/GiB < free}")

# The number that decides whether the levers matter: what is on the card that is NOT params.
alloc = torch.cuda.memory_allocated()
print(f"\nallocated {alloc/GiB:.2f} GiB = params {on_card/GiB:.2f} + "
      f"other {(alloc-on_card)/GiB:.2f} (KV pool, state pool, workspaces)")

# The ten largest card tensors by name, so a surprise has a name rather than a bucket.
big = sorted(((v.numel() * v.element_size(), k, v) for k, v in model.params.items()
              if str(v.device).startswith("cuda")), reverse=True)[:10]
print("\nten largest card params:")
for b, k, v in big:
    print(f"  {k:44} {str(tuple(v.shape)):>18} {str(v.dtype):16} {b/GiB:6.2f} GiB")
