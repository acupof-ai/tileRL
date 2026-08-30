import sys
sys.path.insert(0, "src")
from tilerl_kernels import registry, kernels_linear

assert ("fp4", "sm70") in registry._REGISTRY, "sm70 fp4 cell missing"
cell = registry._REGISTRY[("fp4", "sm70")]
assert "linear_fp4_gemv" in cell
assert cell["linear_fp4_gemv"] is kernels_linear.make_linear_fp4_gemv_sm70
assert ("bf16", "sm70") in registry._REGISTRY
print("sm70 cell keys:", sorted(cell.keys()))
print("OK: sm70 registered + gemv wired")
