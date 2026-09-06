import { outcome } from "./protocol"
import { newTurn, paint, settle, type Turn } from "./render"
import { ask, socketUrl } from "./transport"

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

const history: Array<{ role: "user" | "assistant"; content: string }> = []
let inFlight = false

const fail = (turn: Turn, cap: number, message: string): void => {
  settle(turn, "empty", cap)
  turn.note.hidden = false
  turn.note.replaceChildren(document.createTextNode(message))
}

const stream = (turn: Turn, cap: number | null): Promise<void> =>
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
      if (f.t === "delta") {
        if (f.reasoning_content !== undefined) turn.reasoning += f.reasoning_content
        if (f.content !== undefined) turn.answer += f.content
        paint(turn)
      } else if (f.t === "done") {
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
        fail(turn, cap ?? 0, f.message)
      }
    },
  )

const submit = async (): Promise<void> => {
  const text = composer.value.trim()
  if (text === "" || inFlight) return
  inFlight = true
  send.disabled = true
  composer.value = ""

  const you = newTurn(log, "user")
  you.answer = text
  paint(you)
  settle(you, "answered", 0)
  history.push({ role: "user", content: text })

  const turn = newTurn(log, "assistant")
  turn.root.classList.add("pending")
  // Empty box = no cap of our own; the server decides what fits.
  const typed = budget.value.trim()
  const cap = typed === "" ? null : Math.max(1, Number(typed) || 1)

  try {
    await stream(turn, cap)
  } catch (e) {
    fail(turn, cap ?? 0, String(e))
  } finally {
    // A stream that ends without a `done` frame -- a dropped connection -- still
    // has to release the composer, or the page is stuck with no error shown.
    inFlight = false
    send.disabled = false
    turn.root.classList.remove("pending")
    composer.focus()
  }
}

send.addEventListener("click", () => void submit())
composer.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    void submit()
  }
})
