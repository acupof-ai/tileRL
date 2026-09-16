/** Smooth a token stream into per-frame character reveals.
 *
 * Sparse V100 decode delivers one multi-character token every ~110-167 ms
 * (6-9 tok/s). Appending each token as it arrives makes the text jump in
 * token-sized lumps at 6-9 Hz. The existing rAF paint coalescing cannot batch
 * this: a token arrives more slowly than a frame, so every token still paints.
 *
 * Deltas go into a queue; a scheduler (requestAnimationFrame in the page) drains
 * a few characters per frame, turning the 110 ms arrival granularity into a
 * ~16 ms visual one. The drain step grows with the backlog so a fast model or a
 * burst catches up instead of falling ever further behind, but it is capped so
 * catch-up never re-introduces a visible multi-character jump.
 *
 * Pure logic: the scheduler/canceller are injected, so the timing tests run with
 * a manual step function and no wall clock.
 */

/** Characters revealed on a frame with an empty backlog — the floor rate. */
export const BASE_CHARS = 1
/** Most characters one frame may reveal, however large the backlog. */
export const MAX_CHARS = 8
/** A character is added to the per-frame step for each this many queued
 * characters: queue 8 → +1/frame, queue 56 → the MAX_CHARS cap. */
const BACKLOG_DIVISOR = 8

export interface Reveal {
  /** Queue a freshly arrived chunk; it will leave via onReveal over frames. */
  push: (chunk: string) => void
  /** Reveal every queued character synchronously, once. Terminal/stop/pagehide. */
  flush: () => void
  /** Characters received but not yet revealed. */
  readonly pendingLength: number
}

export interface Timers {
  schedule: (cb: () => void) => number
  cancel: (handle: number) => void
}

export const createReveal = (
  onReveal: (chunk: string) => void,
  timers: Timers = {
    schedule: (cb) => requestAnimationFrame(cb),
    cancel: (h) => cancelAnimationFrame(h),
  },
): Reveal => {
  let pending = ""
  let start = 0
  let handle: number | null = null

  const queued = (): number => pending.length - start

  const tick = (): void => {
    // Floor rate is BASE_CHARS whenever anything is queued; the backlog term
    // only adds whole characters, so a thin stream reveals a steady one per
    // frame with no fractional-credit jitter, and a burst speeds up to the cap.
    const step = Math.min(MAX_CHARS, BASE_CHARS + Math.floor(queued() / BACKLOG_DIVISOR))
    const n = Math.min(step, queued())
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
      // Compact a fully-drained prefix before appending.
      if (start > 0 && start === pending.length) {
        pending = ""
        start = 0
      }
      pending += chunk
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
      if (rest !== "") onReveal(rest)
    },
    get pendingLength(): number {
      return queued()
    },
  }
}
