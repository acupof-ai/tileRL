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
    const ws = new WebSocket(url)
    ws.onopen = () => ws.send(JSON.stringify(body))
    ws.onmessage = (e) => {
      const f = parseFrame(typeof e.data === "string" ? e.data : "")
      // A frame we cannot parse is dropped, not fatal: the stream is still live
      // and the next frame may be fine. Logged so protocol drift is visible
      // rather than silently rendering short.
      if (f === null) console.warn("tilerl: unparseable frame", e.data)
      else {
        if (f.t !== "delta") terminal = true
        onFrame(f)
      }
    }
    ws.onclose = () => resolve(classifyClose(stopped, terminal))
    // A failed handshake fires onerror then onclose; with no terminal frame that
    // classifies as "dropped", which is the honest reason too.
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
