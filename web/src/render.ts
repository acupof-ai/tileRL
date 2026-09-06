/** DOM writing. Everything here takes text and produces nodes; nothing parses the
 * wire format (protocol.ts) and nothing opens a socket (transport.ts). */

/** Markdown, the subset model output actually uses: fenced code, inline code,
 * bold, italic. Built with createTextNode and element nodes rather than
 * innerHTML, so no reply can inject markup -- the previous page carried a
 * hand-written escaper for exactly this and it is a class of bug not worth
 * keeping alive. */
export const markdown = (src: string): DocumentFragment => {
  const frag = document.createDocumentFragment()
  for (const part of src.split(/(```[\s\S]*?(?:```|$))/)) {
    if (part === "") continue
    if (part.startsWith("```")) {
      const body = part.slice(3).replace(/^[^\n]*\n?/, "").replace(/```$/, "")
      const pre = document.createElement("pre")
      const code = document.createElement("code")
      code.appendChild(document.createTextNode(body))
      pre.appendChild(code)
      frag.appendChild(pre)
    } else {
      const p = document.createElement("div")
      p.className = "prose"
      inline(part, p)
      frag.appendChild(p)
    }
  }
  return frag
}

const inline = (src: string, into: HTMLElement): void => {
  // One pass, alternating literal and marked-up runs. `split` with a capturing
  // group keeps the delimiters, so every character lands in exactly one branch
  // and nothing can be dropped by a pattern that fails to match.
  for (const tok of src.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/)) {
    if (tok === "") continue
    if (tok.startsWith("`") && tok.endsWith("`") && tok.length > 1) {
      const c = document.createElement("code")
      c.appendChild(document.createTextNode(tok.slice(1, -1)))
      into.appendChild(c)
    } else if (tok.startsWith("**") && tok.endsWith("**") && tok.length > 3) {
      const b = document.createElement("strong")
      b.appendChild(document.createTextNode(tok.slice(2, -2)))
      into.appendChild(b)
    } else if (tok.startsWith("*") && tok.endsWith("*") && tok.length > 2) {
      const i = document.createElement("em")
      i.appendChild(document.createTextNode(tok.slice(1, -1)))
      into.appendChild(i)
    } else {
      into.appendChild(document.createTextNode(tok))
    }
  }
}

export interface Turn {
  readonly root: HTMLElement
  /** Live text of each half; the DOM is rebuilt from these, never appended to,
   * so a partial UTF-8 tail that later completes cannot leave a stale glyph. */
  reasoning: string
  answer: string
  readonly fold: HTMLDetailsElement
  readonly reasoningBody: HTMLElement
  readonly answerBody: HTMLElement
  readonly note: HTMLElement
}

export const newTurn = (into: HTMLElement, role: "user" | "assistant"): Turn => {
  const root = document.createElement("div")
  root.className = `turn ${role}`

  const fold = document.createElement("details")
  fold.className = "reasoning"
  fold.hidden = true
  const summary = document.createElement("summary")
  summary.appendChild(document.createTextNode("reasoning"))
  fold.appendChild(summary)
  const reasoningBody = document.createElement("div")
  reasoningBody.className = "reasoning-body"
  fold.appendChild(reasoningBody)

  const answerBody = document.createElement("div")
  answerBody.className = "answer"
  const note = document.createElement("div")
  note.className = "note"
  note.hidden = true

  root.append(fold, answerBody, note)
  into.appendChild(root)
  return { root, reasoning: "", answer: "", fold, reasoningBody, answerBody, note }
}

export const paint = (t: Turn): void => {
  if (t.reasoning !== "") {
    t.fold.hidden = false
    t.reasoningBody.replaceChildren(document.createTextNode(t.reasoning))
  }
  t.answerBody.replaceChildren(markdown(t.answer))
}

/** The end state. `truncated` is the only one that writes a notice: the reasoning
 * is already on screen and stays open, so the reader sees what the budget was
 * spent on instead of an empty bubble. */
export const settle = (t: Turn, kind: "answered" | "truncated" | "empty", cap: number): void => {
  t.root.classList.remove("pending")
  if (kind === "truncated") {
    t.fold.open = true
    t.note.hidden = false
    t.note.replaceChildren(
      document.createTextNode(
        `stopped at the ${cap}-token budget while still reasoning — no answer was produced`,
      ),
    )
  } else if (kind === "empty") {
    t.note.hidden = false
    t.note.replaceChildren(document.createTextNode("the model returned an empty reply"))
  }
}
