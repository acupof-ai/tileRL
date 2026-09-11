# Card 5 of the H20 node measures 8% low: a silent 1830 MHz die cap on an otherwise healthy card

2026-09-11 · found calibrating cards 4/5 for the roofline floors

## Context

`bench --calibrate` records per-card `bf16_peak_tflops` from one 8192² bf16
GEMM. Six of the node's H20s (cards 0–4) measure 136.4–137.8 TFLOP/s. Card 5
measured 125.76, reproduced on every run:

| run | card 5 peak (TFLOP/s) |
|---|---:|
| calibrate #1 | 125.76 |
| calibrate #2 | 125.89 |
| calibrate #3 (clock sampling) | 125.66 |
| sustained 20 s GEMM loop | 126.2 (114.7 GEMM/s) |

Card 4 on the same sustained probe: **137.6 TFLOP/s (125.2 GEMM/s)**.

## Root cause

The ratio matches the clock ratio exactly: 126.2 / 137.6 = 0.917, and card 5
holds **1830 MHz** under sustained load while card 4 holds the 1980 MHz
application clock (1830/1980 = 0.924). Nothing visible throttles it:

- power draw 114 W against a 500 W limit (not SW power cap);
- 35 °C, HW thermal / HW slowdown Not Active;
- P state active, Applications Clocks Setting Not Active, SW Thermal and Sync
  Boost Not Active — every `nvidia-smi -q -d PERFORMANCE` reason idle;
- HBM bandwidth is normal: 3312 / 3248 GB/s across runs, same 2619 MHz mem
  clock as the other cards.

The card simply cannot be coaxed past 1830 SM MHz — a die/firmware cap with
no exposed reason. Treating the node as homogeneous is wrong: this card is a
permanently different compute device under the same marketing name
"NVIDIA H20".

The ledger could not represent that: calibration floors keyed on the device
**name** only and took the newest row, so harvesting card 5 would have set the
single H20 compute floor to 125.7 for every card — every later `bench
--kernels` %bound 8% looser, decided by which card happened to be calibrated
last. The visible card index gives no protection (`CUDA_VISIBLE_DEVICES`
renumbers the one visible card to 0, so `device.card` is 0 on every run).

## Fix

Floors carry the physical `device.uuid` (`str(torch.cuda.get_device_properties(card).uuid)`,
verified byte-equal to `nvidia-smi --query-gpu=uuid` through the visibility
mask) and `latest_floor(name, uuid=None)` narrows to that card's own row when
one exists, falling back to the name pool when it does not — so earlier
name-only rows (52/65, cards 0–3) remain valid floors until those cards are
re-calibrated. Card 5's floor is then 125.7 for card 5 only.

## Rule

Same-name GPUs are not the same device. A per-device measured floor needs the
physical identity (UUID, independent of the visible ordinal), and a
population floor needs either a spread-aware key or an explicit conservative
choice — never "newest row for the marketing name". A clock ratio that
matches the performance ratio, with every throttle reason inactive, is the
signature of a die cap; re-running calibration does not clear it.
