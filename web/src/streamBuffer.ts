/** Smooth a token stream into per-frame character reveals.
 *
 * Sparse V100 decode is bursty: graph ticks deliver characters steadily, then
 * one eager refresh stalls the wire for ~218-272 ms on a strict 32-tick period;
 * under concurrency the whole stream runs at half that rate. Revealing each
 * chunk as it arrives exposes the stalls directly, and counting the pace in
 * FRAMES made it depend on the display: at 120 Hz one char/frame is twice the
 * production rate and an 18-frame bank is half the milliseconds.
 *
 * Time-driven instead, so the behaviour is identical at 60/120/144 Hz:
 *   1. Headroom bank — the first characters are held HEADROOM_MS after the
 *      first character arrives, so a reserve exists before the first refresh.
 *   2. Steady drain — each frame earns `credit = rate * dt`, where rate is a
 *      fixed fraction (RATE_FRACTION) of the cumulative MEAN arrival rate
 *      (arrived chars / wall since first char). Revealing slower than the mean
 *      banks the (1-f) difference during fill cycles and spends it during the
 *      periodic stalls: the stall duty cycle (~18%) is what sets the fraction,
 *      and a uniform time stretch (a slower concurrent regime) leaves that
 *      ratio unchanged, so no retune is needed.
 *   3. Bounded catch-up — only once the backlog reaches rate * CATCHUP_MS does
 *      an extra bounded term drain it, capped at MAX_CHARS/frame. Insurance for
 *      a genuine regime change; ordinary stall absorption never reaches it.
 *   4. flush() reveals everything at once (done/stop/pagehide); the terminal
 *      tail never waits.
 *
 * Pure logic. The frame scheduler is injected; its callback receives the frame
 * timestamp (requestAnimationFrame already passes a DOMHighResTimeStamp), so
 * timing tests replay a recorded arrival timeline on a virtual clock with no
 * wall time.
 */

/** Hold the first characters this many ms after the first character arrives. */
export const HEADROOM_MS = 300
/** Reveal at this fraction of the cumulative mean arrival rate; the remainder
 * banks during fill cycles and covers the periodic refresh stall. */
const RATE_FRACTION = 0.8
/** A backlog worth this many ms of production adds a bounded catch-up term. */
const CATCHUP_MS = 600
/** A tab-switch rAF pause must not bank infinite credit; cap per-frame dt. */
const DT_CAP_MS = 100
/** Most characters one frame may reveal, however large the credit/backlog. */
export const MAX_CHARS = 8

export interface Reveal {
  /** Queue a freshly arrived chunk; it leaves via onReveal over frames. */
  push: (chunk: string) => void
  /** Reveal every queued character synchronously, once. Terminal/stop/pagehide. */
  flush: () => void
  /** Characters received but not yet revealed. */
  readonly pendingLength: number
}

export interface Timers {
  /** Run cb on the next frame, passing it the frame timestamp in ms. */
  schedule: (cb: (now: number) => void) => number
  cancel: (handle: number) => void
  /** Monotonic ms clock for arrival timestamps; tests inject a virtual one. */
  now: () => number
}

export const createReveal = (
  onReveal: (chunk: string) => void,
  timers: Timers = {
    schedule: (cb) => requestAnimationFrame((t) => cb(t)),
    cancel: (h) => cancelAnimationFrame(h),
    now: () => performance.now(),
  },
): Reveal => {
  let pending = ""
  let start = 0
  let handle: number | null = null

  let banking = true
  /** Wall time of the first queued character; the bank and rate anchor here. */
  let firstAt: number | null = null
  /** Total characters that have ever arrived (never decremented). */
  let arrived = 0
  /** Fractional reveal credit in characters. */
  let credit = 0
  let lastTick: number

  const queued = (): number => pending.length - start

  const tick = (now: number): void => {
    if (banking) {
      // Still buying headroom: keep the loop alive but reveal nothing.
      if (queued() === 0) {
        handle = null
        return
      }
      if (now - (firstAt as number) < HEADROOM_MS) {
        handle = timers.schedule(tick)
        return
      }
      banking = false
      lastTick = now // credit accrues only AFTER the bank opens
    }

    const dt = Math.min(now - lastTick, DT_CAP_MS)
    lastTick = now
    const mean = arrived / (now - (firstAt as number)) // chars/ms, stalls included
    const rate = RATE_FRACTION * mean
    credit += rate * dt
    const backlog = queued()
    if (backlog >= rate * CATCHUP_MS) {
      credit += (backlog * dt) / CATCHUP_MS
    }
    const n = Math.min(MAX_CHARS, Math.floor(credit), backlog)
    credit -= n
    if (n > 0) {
      onReveal(pending.slice(start, start + n))
      start += n
    }
    if (queued() === 0) {
      pending = ""
      start = 0
      handle = null
      return
    }
    handle = timers.schedule(tick)
  }

  const kick = (): void => {
    if (handle === null && queued() > 0) handle = timers.schedule(tick)
  }

  return {
    push(chunk: string): void {
      if (chunk === "") return
      if (start > 0 && start === pending.length) {
        pending = ""
        start = 0
      }
      pending += chunk
      arrived += chunk.length
      if (banking && firstAt === null) firstAt = timers.now()
      kick()
    },
    flush(): void {
      if (handle !== null) {
        timers.cancel(handle)
        handle = null
      }
      const rest = queued() > 0 ? pending.slice(start) : ""
      pending = ""
      start = 0
      banking = true
      firstAt = null
      arrived = 0
      credit = 0
      if (rest !== "") onReveal(rest)
    },
    get pendingLength(): number {
      return queued()
    },
  }
}
