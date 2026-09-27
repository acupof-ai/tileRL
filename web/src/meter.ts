/** Live speed readout for one turn.

 * TTFT is the wall time from submit to the first delta; the streaming rate is
 * the token throughput over a trailing ~2 s window of the server's cumulative
 * token counts (delta frames carry `tokens`), falling back to the cumulative
 * mean until the window spans two samples. The terminal line reports the
 * whole-turn mean = completion_tokens / (done time − first-frame time).
 *
 * Pure logic; every method takes the clock reading, so the gate feeds a
 * synthetic frame sequence on a fake clock and reads the exact text.
 */

/** Window for the live tok/s reading. */
const WINDOW_MS = 2000

export interface SpeedMeter {
  /** Turn submitted; starts the TTFT clock. */
  start: (now: number) => void
  /** Observe the cumulative generated token count from one delta frame. The
   * first call marks the first-frame time. */
  observe: (tokens: number, now: number) => void
  /** Live meter text ("" before the first frame). */
  liveText: (now: number) => string
  /** Terminal line for the done frame. */
  finishText: (promptTokens: number, completionTokens: number, now: number) => string
}

export const createSpeedMeter = (): SpeedMeter => {
  let submittedAt: number | null = null
  let firstAt: number | null = null
  const samples: Array<{ t: number; tokens: number }> = []

  const liveRate = (now: number): number | null => {
    if (firstAt === null) return null
    const recent = samples.filter((s) => s.t >= now - WINDOW_MS)
    const a = recent[0]
    const b = recent[recent.length - 1]
    if (a !== undefined && b !== undefined && b.t > a.t) {
      return ((b.tokens - a.tokens) / (b.t - a.t)) * 1000
    }
    // Not enough window yet: quote the cumulative mean since the first frame.
    if (b !== undefined && now > firstAt) return (b.tokens / (now - firstAt)) * 1000
    return null
  }

  return {
    start(now: number): void {
      submittedAt = now
    },
    observe(tokens: number, now: number): void {
      if (firstAt === null) firstAt = now
      samples.push({ t: now, tokens })
    },
    liveText(now: number): string {
      if (firstAt === null || submittedAt === null) return ""
      const parts = [`TTFT ${((firstAt - submittedAt) / 1000).toFixed(2)} s`]
      const rate = liveRate(now)
      if (rate !== null && Number.isFinite(rate) && rate > 0) parts.push(`${rate.toFixed(1)} tok/s`)
      return parts.join(" · ")
    },
    finishText(promptTokens: number, completionTokens: number, now: number): string {
      const ttft = firstAt === null || submittedAt === null ? 0 : (firstAt - submittedAt) / 1000
      const dt = firstAt === null ? 0 : (now - firstAt) / 1000
      const rate = dt > 0 ? completionTokens / dt : 0
      return (
        `${promptTokens} prompt + ${completionTokens} completion · ` +
        `${rate.toFixed(1)} tok/s · TTFT ${ttft.toFixed(2)} s`
      )
    },
  }
}
