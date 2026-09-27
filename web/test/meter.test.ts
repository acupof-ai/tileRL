import assert from "node:assert/strict"
import { test } from "node:test"

import { createSpeedMeter } from "../src/meter.ts"
import { parseFrame } from "../src/protocol.ts"

test("the speed meter derives TTFT, a 2 s-window tok/s, and a whole-turn mean", () => {
  const m = createSpeedMeter()
  // Submit at t=0; prefill takes 500 ms, so the first delta (1 generated token)
  // sets TTFT to 0.50 s. Tokens then arrive with cumulative counts.
  m.start(0)
  m.observe(1, 500)
  assert.equal(m.liveText(500), "TTFT 0.50 s", "no rate from a single point")
  // 79 more tokens over the next 2.0 s: the trailing window spans t=500..2500.
  m.observe(20, 1000)
  m.observe(40, 1500)
  m.observe(80, 2500)
  const live = m.liveText(2500)
  assert.match(live, /TTFT 0\.50 s/)
  assert.match(live, /39\.5 tok\/s/, live) // (80-1) tokens / 2.0 s
  // The server's additive `tokens` field reaches the meter through the parser.
  const f = parseFrame('{"t":"delta","content":"hi","tokens":80}')
  assert.equal(f?.t, "delta")
  assert.equal("tokens" in f && f.tokens, 80)
  // done at t=2500: 80 completion tokens over 2.0 s after the first frame.
  assert.equal(
    m.finishText(120, 80, 2500),
    "120 prompt + 80 completion · 40.0 tok/s · TTFT 0.50 s",
  )
})

test("a delta frame without tokens (old server) still parses", () => {
  const f = parseFrame('{"t":"delta","content":"hi"}')
  assert.equal(f?.t, "delta")
  assert.equal("tokens" in f, false)
})
