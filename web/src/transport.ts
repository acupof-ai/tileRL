import { type CloseKind, type Frame, classifyClose, parseFrame } from "./protocol.ts"

/** One request, one socket, closed when the stream ends.
 *
 * A socket per turn rather than a persistent one: the server holds a single
 * engine request per connection, so a shared socket would need request ids and a
 * demultiplexer for a page that never has two turns in flight.
 *
 * Resolves with why the socket closed (see `classifyClose`): a user stop, a
 * close after a terminal frame, or an abnormal mid-turn drop. There is no
 * auto-reconnect: the protocol carries no frame ids, so a resend regenerates
 * the turn and every streamed token would appear twice. The caller decides
 * whether a `dropped` turn offers a manual retry.
 *
 * `onStop` receives a function that closes the socket. Handing it out rather than
 * returning the socket keeps the WebSocket itself inside this file -- the caller
 * can end the stream and cannot send on it.
 */
export const ask = (
  url: string,
  body: unknown,
  onFrame: (f: Frame) => void,
  onStop?: (stop: () => void) => void,
): Promise<CloseKind> =>
  new Promise<CloseKind>((resolve) => {
    let stopped = false
    let terminal = false
    let opened = false
    const ws = new WebSocket(url)
    ws.onopen = () => {
      opened = true
      ws.send(JSON.stringify(body))
    }
    ws.onmessage = (e) => {
      const f = parseFrame(typeof e.data === "string" ? e.data : "")
      // A frame we cannot parse is dropped, not fatal: the stream is still live
      // and the next frame may be fine. Logged so protocol drift is visible
      // rather than silently rendering short.
      if (f === null) console.warn("tilerl: unparseable frame", e.data)
      else {
        // Only a terminal frame (done/error) ends the classification. A
        // tool_calls frame is NOT terminal — it precedes done — so a close after
        // it but before done must still read as a drop, not a clean finish.
        terminal = f.t === "done" || f.t === "error"
        onFrame(f)
      }
    }
    ws.onclose = () => resolve(classifyClose(stopped, terminal, opened))
    // A failed handshake (server down or restarting) fires onerror then onclose
    // with onopen never having run: that is "unreachable", distinct from a
    // mid-turn drop.
    ws.onerror = () => {}
    // Closing the socket cancels the request server-side.
    onStop?.(() => {
      stopped = true
      ws.close()
    })
  })

/** ws:// for http://, wss:// for https://. Derived from the page's own origin so
 * the bundle carries no host and works behind whatever the pod puts in front. */
export const socketUrl = (loc: Location, path: string): string =>
  `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}${path}`

/** Poll /health until it answers 200, backing off across a supervisor restart
 * (a reload takes tens of seconds). Resolves true when the server is back;
 * false on timeout; the caller supplies the timers so tests run them
 * synchronously. An abort function is handed to onStop so Stop ends the wait.
 */
export const waitForHealth = async (
  healthUrl: string,
  onTick: (attempt: number, delayMs: number) => void,
  timers: { sleep: (ms: number) => Promise<void>; now: () => number },
  onStop?: (abort: () => void) => void,
): Promise<boolean> => {
  const MAX_MS = 180_000
  const start = timers.now()
  let attempt = 0
  let aborted = false
  onStop?.(() => {
    aborted = true
  })
  for (;;) {
    if (aborted) return false
    attempt += 1
    // 1,2,4… capped at 16s.
    const delayMs = Math.min(16_000, 500 * 2 ** (attempt - 1))
    try {
      const r = await fetch(healthUrl, { cache: "no-store" })
      if (r.ok) return true
    } catch {
      // connection refused during the reload window
    }
    if (timers.now() - start > MAX_MS) return false
    onTick(attempt, delayMs)
    await timers.sleep(delayMs)
  }
}
