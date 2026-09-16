/** Frames on the wire. The server owns the reasoning/answer split (`split_think`
 * in prompt.py); this file only names what arrives, so the two sides cannot
 * disagree about where a `</think>` goes. */

/** Sent once, when the composer submits. */
export interface Ask {
  readonly messages: ReadonlyArray<{ role: "user" | "assistant"; content: string }>
  readonly max_tokens?: number
  readonly enable_thinking?: boolean
}

export interface ToolCall {
  readonly id: string
  readonly name: string
  readonly arguments: string
}

/** `reasoning_content` and `content` are the SSE route's field names (#159). Same
 * names here so a reader of either transport learns one vocabulary.
 *
 * `tool_calls` is additive: emitted once before the terminal frame when the
 * model asked for a tool. */
export type Frame =
  | { readonly t: "delta"; readonly reasoning_content?: string; readonly content?: string }
  | { readonly t: "tool_calls"; readonly tool_calls: ReadonlyArray<ToolCall> }
  | {
      readonly t: "done"
      readonly finish_reason: string
      readonly tool_calls?: ReadonlyArray<ToolCall>
      readonly usage: Usage
    }
  | { readonly t: "error"; readonly message: string }

export interface Usage {
  readonly prompt_tokens: number
  readonly completion_tokens: number
}

const asToolCalls = (v: unknown): ReadonlyArray<ToolCall> | null => {
  if (!Array.isArray(v)) return null
  const out: ToolCall[] = []
  for (const item of v) {
    if (typeof item !== "object" || item === null) return null
    const o = item as Record<string, unknown>
    if (typeof o["id"] !== "string" || typeof o["name"] !== "string") return null
    if (typeof o["arguments"] !== "string") return null
    out.push({ id: o["id"], name: o["name"], arguments: o["arguments"] })
  }
  return out
}

/** A frame off the socket is untrusted input: it arrives as text and is parsed
 * before anything renders it. Returning null rather than throwing keeps a bad
 * frame from tearing down a live stream. */
export const parseFrame = (raw: string): Frame | null => {
  let v: unknown
  try {
    v = JSON.parse(raw)
  } catch {
    return null
  }
  if (typeof v !== "object" || v === null) return null
  const o = v as Record<string, unknown>
  if (o["t"] === "delta") {
    const r = o["reasoning_content"]
    const c = o["content"]
    if (r !== undefined && typeof r !== "string") return null
    if (c !== undefined && typeof c !== "string") return null
    return {
      t: "delta",
      ...(typeof r === "string" ? { reasoning_content: r } : {}),
      ...(typeof c === "string" ? { content: c } : {}),
    }
  }
  if (o["t"] === "tool_calls") {
    const calls = asToolCalls(o["tool_calls"])
    if (calls === null) return null
    return { t: "tool_calls", tool_calls: calls }
  }
  if (o["t"] === "done") {
    const u = o["usage"]
    if (typeof o["finish_reason"] !== "string" || typeof u !== "object" || u === null) return null
    const uo = u as Record<string, unknown>
    if (typeof uo["prompt_tokens"] !== "number" || typeof uo["completion_tokens"] !== "number") {
      return null
    }
    const calls = o["tool_calls"] === undefined ? undefined : asToolCalls(o["tool_calls"])
    if (o["tool_calls"] !== undefined && calls === null) return null
    return {
      t: "done",
      finish_reason: o["finish_reason"],
      ...(calls ? { tool_calls: calls } : {}),
      usage: { prompt_tokens: uo["prompt_tokens"], completion_tokens: uo["completion_tokens"] },
    }
  }
  if (o["t"] === "error" && typeof o["message"] === "string") {
    return { t: "error", message: o["message"] }
  }
  return null
}

/** Why a stream's socket closed. The close code alone cannot say it: the server
 * ends a finished turn with 1000 and a user stop also sends 1000, and a 1001
 * after a `done` frame is a normal post-terminal shutdown, not a dropped turn.
 * `unreachable` means the socket never opened — the server is down or restarting
 * (the supervisor reloads for tens of seconds), so the caller polls /health
 * before offering a retry rather than failing a click instantly. Track the three
 * facts that distinguish the cases. */
export type CloseKind = "stopped" | "terminal" | "dropped" | "unreachable"

export const classifyClose = (
  stopped: boolean,
  terminal: boolean,
  opened: boolean,
): CloseKind => {
  if (stopped) return "stopped"
  if (terminal) return "terminal"
  return opened ? "dropped" : "unreachable"
}

/** What the page shows once the stream ends.
 *
 * The state ckl hit: thinking on, the 512-token default budget spent inside the
 * block, so the reply is reasoning only and the answer is empty. `truncated` is
 * that case and nothing else -- an empty answer that finished normally is a model
 * that chose to say nothing, which is not the same bug and should not carry the
 * same notice. `error` is an in-band `error` frame: the server refused or failed
 * the request, which is not an "empty reply" even when zero tokens arrived. */
export const outcome = (
  finish: string,
  answer: string,
  sawReasoning: boolean,
): "answered" | "truncated" | "empty" => {
  if (answer.length > 0) return "answered"
  return finish === "length" && sawReasoning ? "truncated" : "empty"
}
