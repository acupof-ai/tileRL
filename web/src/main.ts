import { outcome } from "./protocol.ts"
import { newTurn, paint, pruneTurns, settle, type Turn } from "./render.ts"
import { ask, socketUrl } from "./transport.ts"

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id)
  if (el === null) throw new Error(`missing element #${id}`)
  return el as T
}

const log = $("log")
const composer = $<HTMLTextAreaElement>("composer")
const send = $<HTMLButtonElement>("send")
const thinking = $<HTMLInputElement>("thinking")
const budget = $<HTMLInputElement>("budget")
const meter = $("meter")
const stop = $<HTMLButtonElement>("stop")
const toBottom = $<HTMLButtonElement>("to-bottom")

/** Within a few px of the bottom. Not `=== 0`: sub-pixel layout and zoom leave a
 * fractional remainder on a log the reader has scrolled all the way down. */
const atBottom = (el: HTMLElement): boolean =>
  el.scrollHeight - el.scrollTop - el.clientHeight < 32

/** Render at most once per display frame. Frames can arrive far faster than the
 * screen paints, and every paint re-lexes the block still streaming; a 128k
 * reply repainted per token is the page freeze. Whether to follow the stream is
 * measured at PAINT time (before the nodes change): measuring when the frame
 * arrived would decide against the scroll state the reader is looking at. */
let frameQueued = false
let paintingTurn: Turn | null = null
const schedulePaint = (turn: Turn): void => {
  paintingTurn = turn
  if (frameQueued) return
  frameQueued = true
  requestAnimationFrame(() => {
    frameQueued = false
    const t = paintingTurn
    paintingTurn = null
    if (t === null) return
    // A stop or drop settles the turn while a paint is still queued; a `done`
    // frame paints synchronously before releasing pending. Either way a late
    // frame only churns a finished DOM.
    if (t.root.classList.contains("pending") === false) return
    const stick = atBottom(log)
    paint(t)
    if (stick) log.scrollTop = log.scrollHeight
  })
}

/** A long session is a long DOM: cap the rendered log. Conversation history is
 * text and stays intact; the in-flight turn and the freshest reply are exempt
 * inside pruneTurns. A "clear history" affordance is a later change. */
const MAX_TURNS = 40

const history: Array<{ role: "user" | "assistant"; content: string }> = []
let inFlight = false

const note = (turn: Turn, message: string, retry?: () => void): void => {
  turn.note.hidden = false
  turn.note.replaceChildren(document.createTextNode(message))
  if (retry !== undefined) {
    const b = document.createElement("button")
    b.className = "ghost retry"
    b.appendChild(document.createTextNode("Retry"))
    b.addEventListener("click", retry)
    turn.note.appendChild(b)
  }
}

const fail = (turn: Turn, cap: number, message: string): void => {
  settle(turn, "empty", cap)
  note(turn, message)
}

let stopStream: (() => void) | null = null

const stream = (turn: Turn, cap: number | null, resend: () => void): Promise<void> =>
  ask(
    socketUrl(window.location, "/ws/chat"),
    {
      messages: history,
      // Omitted, not zero: an empty box means "whatever fits", and the server owns
      // that number. Sending a placeholder here would put the page's guess in
      // front of the context remainder the server computes.
      ...(cap === null ? {} : { max_tokens: cap }),
      enable_thinking: thinking.checked,
    },
    (f) => {
      // A terminal frame ends the turn; anything after it on this socket belongs
      // to a connection the server is already shutting down.
      if (turn.root.classList.contains("final")) return
      if (f.t === "delta") {
        if (f.reasoning_content !== undefined) turn.reasoning += f.reasoning_content
        if (f.content !== undefined) turn.answer += f.content
        schedulePaint(turn)
      } else if (f.t === "done") {
        turn.root.classList.add("final")
        // Flush the coalesced paint before settling, or the last tokens can miss
        // the DOM of a turn that is already marked finished.
        paint(turn)
        // The notice names the budget that was actually spent, so with no typed cap
        // it comes from usage -- the page has no other honest number to quote.
        settle(turn, outcome(f.finish_reason, turn.answer, turn.reasoning !== ""),
               cap ?? f.usage.completion_tokens)
        meter.replaceChildren(
          document.createTextNode(
            `${f.usage.prompt_tokens} prompt + ${f.usage.completion_tokens} completion tokens`,
          ),
        )
        // Only a real answer joins the history. Replaying reasoning as an
        // assistant turn would feed the block back into the next prompt, and
        // replaying an empty answer teaches the model to answer nothing.
        if (turn.answer !== "") history.push({ role: "assistant", content: turn.answer })
      } else {
        turn.root.classList.add("final")
        fail(turn, cap ?? 0, f.message)
      }
    },
    (close) => {
      stopStream = close
    },
  ).then((kind) => {
    if (kind === "dropped") {
      settle(turn, "dropped", cap ?? 0)
      note(turn, "connection lost before the reply finished — retry?", resend)
    } else if (kind === "stopped") {
      settle(turn, "stopped", cap ?? 0)
      // A `done` frame that crossed the close in flight already pushed the
      // answer; the final guard is what keeps it out of history twice.
      if (turn.answer !== "" && turn.root.classList.contains("final") === false) {
        history.push({ role: "assistant", content: turn.answer })
      }
    }
  })

const submit = async (text?: string): Promise<void> => {
  const typed0 = text ?? composer.value.trim()
  if (typed0 === "" || inFlight) return
  inFlight = true
  send.disabled = true
  stop.hidden = false
  toBottom.hidden = true
  if (text === undefined) composer.value = ""

  const you = newTurn(log, "user")
  you.answer = typed0
  paint(you)
  settle(you, "answered", 0)
  history.push({ role: "user", content: typed0 })

  const turn = newTurn(log, "assistant")
  turn.root.classList.add("pending")
  // Empty box = no cap of our own; the server decides what fits.
  const typed = budget.value.trim()
  const cap = typed === "" ? null : Math.max(1, Number(typed) || 1)

  // Manual resend after a dropped connection: this turn's user message is the
  // last history entry, and no assistant answer followed it. Pop the message and
  // the dead turn, then submit the same text again. No auto-resend: the protocol
  // has no frame ids, so an automatic reconnect regenerates and duplicates.
  const resend = (): void => {
    if (history.at(-1)?.role === "user") history.pop()
    turn.root.remove()
    void submit(typed0)
  }

  try {
    await stream(turn, cap, resend)
  } catch (e) {
    fail(turn, cap ?? 0, String(e))
  } finally {
    // A stream that ends without a `done` frame -- a dropped connection -- still
    // has to release the composer, or the page is stuck with no error shown.
    inFlight = false
    send.disabled = false
    stop.hidden = true
    stopStream = null
    turn.root.classList.remove("pending")
    pruneTurns(log, MAX_TURNS)
    composer.focus()
  }
}

send.addEventListener("click", () => void submit())
stop.addEventListener("click", () => stopStream?.())
composer.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    void submit()
  }
})

// Shown only while the reader has scrolled away from the live end; clicking it
// restores the follow state, and the next paint sticks.
toBottom.addEventListener("click", () => {
  log.scrollTop = log.scrollHeight
})
log.addEventListener("scroll", () => {
  toBottom.hidden = atBottom(log)
})
