/** DOM writing. Everything here takes text and produces nodes; nothing parses the
 * wire format (protocol.ts) and nothing opens a socket (transport.ts). */

/** Markdown, the subset model output actually uses: headings, paragraphs, lists,
 * links, fenced code, inline code, bold, italic. Built with createTextNode and
 * element nodes rather than innerHTML, so no reply can inject markup -- the page
 * once carried a hand-written escaper for exactly this and it is a class of bug
 * not worth keeping alive.
 *
 * No parser library: the whole grammar below is ~60 lines and marked/markdown-it
 * are 30-40 KB gzipped against a 3.9 KB page. */
export const markdown = (src: string): DocumentFragment => {
  const frag = document.createDocumentFragment()
  // Fences first, so a `#` or `-` inside a code block is never a heading or a
  // bullet. `(?:```|$)` is what makes a half-arrived block render while it
  // streams: an unterminated fence is a code block whose body is what has come
  // in so far, not a paragraph that turns into one when the closer lands.
  for (const part of src.split(/(```[\s\S]*?(?:```|$))/)) {
    if (part === "") continue
    if (part.startsWith("```")) frag.appendChild(fence(part))
    else blocks(part, frag)
  }
  return frag
}

const fence = (part: string): HTMLElement => {
  const nl = part.indexOf("\n")
  const body = (nl === -1 ? "" : part.slice(nl + 1)).replace(/```$/, "")
  const pre = document.createElement("pre")
  const code = document.createElement("code")
  code.appendChild(document.createTextNode(body))
  const copy = document.createElement("button")
  copy.className = "copy"
  copy.appendChild(document.createTextNode("Copy"))
  // Outside <code>, so selecting the block or reading its textContent never
  // picks up the word "Copy".
  copy.addEventListener("click", () => {
    void navigator.clipboard?.writeText(body)
    copy.replaceChildren(document.createTextNode("Copied"))
  })
  pre.append(copy, code)
  return pre
}

/** A bullet or a number, and the space after it. The trailing `\s+` is what
 * keeps `*italic*` at the start of a line from reading as a list item. */
const ITEM = /^\s*(?:[-*+]|\d+[.)])\s+/

const blocks = (src: string, into: DocumentFragment): void => {
  const lines = src.split("\n")
  let i = 0
  while (i < lines.length) {
    const line = lines[i] ?? ""
    if (line.trim() === "") {
      i++
      continue
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line)
    if (h !== null) {
      const el = document.createElement(`h${(h[1] ?? "").length}`)
      inline(h[2] ?? "", el)
      into.appendChild(el)
      i++
      continue
    }
    if (ITEM.test(line)) {
      const list = document.createElement(/^\s*\d/.test(line) ? "ol" : "ul")
      while (i < lines.length && ITEM.test(lines[i] ?? "")) {
        const li = document.createElement("li")
        inline((lines[i] ?? "").replace(ITEM, ""), li)
        list.appendChild(li)
        i++
      }
      into.appendChild(list)
      continue
    }
    const buf: string[] = []
    while (i < lines.length) {
      const l = lines[i] ?? ""
      if (l.trim() === "" || /^#{1,6}\s/.test(l) || ITEM.test(l)) break
      buf.push(l)
      i++
    }
    const p = document.createElement("p")
    p.className = "prose"
    inline(buf.join("\n"), p)
    into.appendChild(p)
  }
}

/** A URL we are willing to put in an href, or null.
 *
 * createElement closes the attribute-breakout hole the old string renderer had,
 * but it does NOT close this one: `[click](javascript:...)` is a live handler
 * however the node was built. Allow-list the three schemes a reply has any
 * business using, plus site-relative, and render anything else as plain text. */
const safeHref = (u: string): string | null => {
  const s = u.trim()
  return /^(?:https?:\/\/|mailto:)/i.test(s) || /^[/#]/.test(s) ? s : null
}

const inline = (src: string, into: HTMLElement): void => {
  // One pass, alternating literal and marked-up runs. `split` with a capturing
  // group keeps the delimiters, so every character lands in exactly one branch
  // and nothing can be dropped by a pattern that fails to match.
  for (const tok of src.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]*\]\([^)\s]*\))/)) {
    if (tok === "") continue
    const link = /^\[([^\]]*)\]\(([^)\s]*)\)$/.exec(tok)
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
    } else if (link !== null) {
      const href = safeHref(link[2] ?? "")
      if (href === null) {
        into.appendChild(document.createTextNode(tok))
      } else {
        const a = document.createElement("a")
        a.href = href
        a.target = "_blank"
        a.rel = "noopener noreferrer"
        a.appendChild(document.createTextNode(link[1] ?? ""))
        into.appendChild(a)
      }
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
