/** Frames on the wire. The server owns the reasoning/answer split (`split_think`
 * in prompt.py); this file only names what arrives, so the two sides cannot
 * disagree about where a `</think>` goes. */

/** Sent once, when the composer submits. */
export interface Ask {
  readonly messages: ReadonlyArray<{ role: "user" | "assistant"; content: string }>
  readonly max_tokens?: number
  readonly enable_thinking?: boolean
}

/** `reasoning_content` and `content` are the SSE route's field names (#159). Same
 * names here so a reader of either transport learns one vocabulary. */
export type Frame =
  | { readonly t: "delta"; readonly reasoning_content?: string; readonly content?: string }
  | { readonly t: "done"; readonly finish_reason: string; readonly usage: Usage }
  | { readonly t: "error"; readonly message: string }

export interface Usage {
  readonly prompt_tokens: number
  readonly completion_tokens: number
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
  if (o["t"] === "done") {
    const u = o["usage"]
    if (typeof o["finish_reason"] !== "string" || typeof u !== "object" || u === null) return null
    const uo = u as Record<string, unknown>
    if (typeof uo["prompt_tokens"] !== "number" || typeof uo["completion_tokens"] !== "number") {
      return null
    }
    return {
      t: "done",
      finish_reason: o["finish_reason"],
      usage: { prompt_tokens: uo["prompt_tokens"], completion_tokens: uo["completion_tokens"] },
    }
  }
  if (o["t"] === "error" && typeof o["message"] === "string") {
    return { t: "error", message: o["message"] }
  }
  return null
}

/** What the page shows once the stream ends.
 *
 * The state ckl hit: thinking on, the 512-token default budget spent inside the
 * block, so the reply is reasoning only and the answer is empty. `truncated` is
 * that case and nothing else -- an empty answer that finished normally is a model
 * that chose to say nothing, which is not the same bug and should not carry the
 * same notice. */
export const outcome = (
  finish: string,
  answer: string,
  sawReasoning: boolean,
): "answered" | "truncated" | "empty" => {
  if (answer.length > 0) return "answered"
  return finish === "length" && sawReasoning ? "truncated" : "empty"
}
