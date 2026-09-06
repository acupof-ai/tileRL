import { type Frame, parseFrame } from "./protocol"

/** One request, one socket, closed when the stream ends.
 *
 * A socket per turn rather than a persistent one: the server holds a single
 * engine request per connection, so a shared socket would need request ids and a
 * demultiplexer for a page that never has two turns in flight.
 *
 * `onFrame` is called for each frame; the promise settles when the socket closes,
 * and rejects only on a transport error. The caller's `finally` is what releases
 * the composer, so a dropped connection cannot leave the page stuck.
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
): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    const ws = new WebSocket(url)
    ws.onopen = () => ws.send(JSON.stringify(body))
    ws.onmessage = (e) => {
      const f = parseFrame(typeof e.data === "string" ? e.data : "")
      // A frame we cannot parse is dropped, not fatal: the stream is still live
      // and the next frame may be fine. Logged so protocol drift is visible
      // rather than silently rendering short.
      if (f === null) console.warn("tilerl: unparseable frame", e.data)
      else onFrame(f)
    }
    // onclose fires for a clean close too, so resolve there rather than waiting
    // on a `done` frame that a dropped connection never sends.
    ws.onclose = () => resolve()
    ws.onerror = () => reject(new Error("connection failed"))
    // resolve on user stop: the tokens on screen are the reply, not a failure
    onStop?.(() => {
      ws.close()
      resolve()
    })
  })

/** ws:// for http://, wss:// for https://. Derived from the page's own origin so
 * the bundle carries no host and works behind whatever the pod puts in front. */
export const socketUrl = (loc: Location, path: string): string =>
  `${loc.protocol === "https:" ? "wss:" : "ws:"}//${loc.host}${path}`
