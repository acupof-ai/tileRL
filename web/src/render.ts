/** DOM writing. Everything here takes text and produces nodes; nothing parses the
 * wire format (protocol.ts) and nothing opens a socket (transport.ts). */

import { marked } from "marked"
import type { Token, Tokens } from "marked"

/** Markdown, GFM, rendered as NODES.
 *
 * `marked` does the grammar -- the hand-rolled version here covered headings, flat
 * lists, fences, links and bold, so a table arrived as prose full of pipes and a
 * nested list flattened (ckl, on the V100 page: "md 组件不全"). What it does NOT do
 * is produce the HTML string: we walk its token tree and build nodes with
 * createElement/createTextNode, so no reply can inject markup. marked.parse() +
 * DOMPurify + innerHTML is the usual shape and costs 23.5 KB gz against 12.8 for
 * the lexer alone -- and it would put an innerHTML sink back in the page, where the
 * old string renderer's attribute breakout lived. Absent beats sanitised. */
export const markdown = (src: string): DocumentFragment => {
  const frag = document.createDocumentFragment()
  // An unterminated fence must render as a code block whose body is what has
  // arrived, not as a paragraph that becomes one when the closer lands; marked's
  // lexer already ends an open fence at EOF, which is the streaming behaviour the
  // hand-rolled splitter was written for.
  for (const tok of marked.lexer(src, { gfm: true })) block(tok, frag)
  return frag
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K): HTMLElementTagNameMap[K] =>
  document.createElement(tag)

const block = (tok: Token, into: Node): void => {
  switch (tok.type) {
    case "space":
      return
    case "heading": {
      const h = document.createElement(`h${(tok as Tokens.Heading).depth}`)
      inline((tok as Tokens.Heading).tokens, h)
      into.appendChild(h)
      return
    }
    case "code":
      into.appendChild(fence(tok as Tokens.Code))
      return
    case "table":
      into.appendChild(table(tok as Tokens.Table))
      return
    case "blockquote": {
      const q = el("blockquote")
      for (const t of (tok as Tokens.Blockquote).tokens) block(t, q)
      into.appendChild(q)
      return
    }
    case "list":
      into.appendChild(list(tok as Tokens.List))
      return
    case "hr":
      into.appendChild(el("hr"))
      return
    case "paragraph": {
      const p = el("p")
      p.className = "prose"
      inline((tok as Tokens.Paragraph).tokens, p)
      into.appendChild(p)
      return
    }
    default: {
      // text, html, def and anything a future marked adds: rendered as its own
      // source text. `html` lands here deliberately -- a reply's raw markup is text.
      const raw = (tok as { text?: string; raw?: string }).text ?? tok.raw ?? ""
      if (raw.trim() === "") return
      const p = el("p")
      p.className = "prose"
      p.appendChild(document.createTextNode(raw))
      into.appendChild(p)
    }
  }
}

const fence = (tok: Tokens.Code): HTMLElement => {
  const body = tok.text
  const pre = el("pre")
  const code = el("code")
  code.appendChild(document.createTextNode(body))
  const copy = el("button")
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

const table = (tok: Tokens.Table): HTMLElement => {
  const t = el("table")
  const thead = el("thead")
  const hr = el("tr")
  tok.header.forEach((cell, i) => {
    const th = el("th")
    const a = tok.align[i]
    if (a) th.style.textAlign = a
    inline(cell.tokens, th)
    hr.appendChild(th)
  })
  thead.appendChild(hr)
  const tbody = el("tbody")
  for (const row of tok.rows) {
    const tr = el("tr")
    row.forEach((cell, i) => {
      const td = el("td")
      const a = tok.align[i]
      if (a) td.style.textAlign = a
      inline(cell.tokens, td)
      tr.appendChild(td)
    })
    tbody.appendChild(tr)
  }
  t.append(thead, tbody)
  return t
}

const list = (tok: Tokens.List): HTMLElement => {
  if (tok.ordered) {
    const ol = el("ol")
    if (tok.start !== "" && tok.start !== 1) ol.start = Number(tok.start)
    return items(tok, ol)
  }
  return items(tok, el("ul"))
}

const items = (tok: Tokens.List, l: HTMLElement): HTMLElement => {
  for (const item of tok.items) {
    const li = el("li")
    if (item.task) {
      const box = el("input")
      box.type = "checkbox"
      box.checked = item.checked === true
      box.disabled = true
      li.appendChild(box)
    }
    // A list item's children are BLOCK tokens (that is how a nested list arrives),
    // but a tight item's text token holds inline children -- those go through
    // inline() rather than being flattened into a paragraph.
    for (const t of item.tokens) {
      if (t.type === "text" && "tokens" in t && Array.isArray(t.tokens)) {
        inline(t.tokens as Token[], li)
      } else {
        block(t, li)
      }
    }
    l.appendChild(li)
  }
  return l
}

/** A URL we are willing to put in an href or src, or null.
 *
 * createElement closes the attribute-breakout hole the old string renderer had,
 * but it does NOT close this one: `[click](javascript:...)` is a live handler
 * however the node was built. Allow-list the three schemes a reply has any
 * business using, plus site-relative, and render anything else as plain text. */
const safeHref = (u: string): string | null => {
  const s = u.trim()
  return /^(?:https?:\/\/|mailto:)/i.test(s) || /^[/#]/.test(s) ? s : null
}

const inline = (toks: Token[], into: HTMLElement): void => {
  for (const tok of toks) {
    switch (tok.type) {
      case "escape":
      case "text":
        // A text token can itself carry inline children (a table cell, a list item).
        if ("tokens" in tok && Array.isArray(tok.tokens)) inline(tok.tokens as Token[], into)
        else into.appendChild(document.createTextNode((tok as Tokens.Text).text))
        break
      case "codespan": {
        const c = el("code")
        c.appendChild(document.createTextNode((tok as Tokens.Codespan).text))
        into.appendChild(c)
        break
      }
      case "strong":
        into.appendChild(wrap("strong", (tok as Tokens.Strong).tokens))
        break
      case "em":
        into.appendChild(wrap("em", (tok as Tokens.Em).tokens))
        break
      case "del":
        into.appendChild(wrap("del", (tok as Tokens.Del).tokens))
        break
      case "br":
        into.appendChild(el("br"))
        break
      case "checkbox":
        break // already rendered by list(), which owns the item's box
      case "link": {
        const lk = tok as Tokens.Link
        const href = safeHref(lk.href)
        if (href === null) {
          into.appendChild(document.createTextNode(lk.raw))
          break
        }
        const a = el("a")
        a.href = href
        a.target = "_blank"
        a.rel = "noopener noreferrer"
        inline(lk.tokens, a)
        into.appendChild(a)
        break
      }
      case "image": {
        const im = tok as Tokens.Image
        const src = safeHref(im.href)
        // Same policy as a link, for the same reason: `src` fetches, and a
        // data:/javascript: URL has no business in a reply.
        if (src === null) {
          into.appendChild(document.createTextNode(im.raw))
          break
        }
        const img = el("img")
        img.src = src
        img.alt = im.text
        if (im.title !== null) img.title = im.title
        into.appendChild(img)
        break
      }
      default:
        into.appendChild(document.createTextNode((tok as { raw: string }).raw))
    }
  }
}

const wrap = (tag: "strong" | "em" | "del", toks: Token[]): HTMLElement => {
  const e = el(tag)
  inline(toks, e)
  return e
}

export interface Turn {
  readonly root: HTMLElement
  /** Live text of each half; the DOM is rebuilt from these, never appended to,
   * so a partial UTF-8 tail that later completes cannot leave a stale glyph. */
  reasoning: string
  answer: string
  /** How much of `answer` is already on screen as finished blocks. Only the text
   * past this point is re-parsed per frame; see `paint`. */
  settled: number
  readonly fold: HTMLDetailsElement
  readonly reasoningBody: HTMLElement
  /** Finished blocks, appended to and never rebuilt. */
  readonly answerBody: HTMLElement
  /** The block still being written; the only node a frame replaces. */
  readonly answerTail: HTMLElement
  readonly note: HTMLElement
}

/** Where the block currently being written begins.
 *
 * Everything before this is finished: no later token can change it, because the
 * grammar's block boundaries are a blank line and a fence, and both are already
 * behind us. Everything after has to be re-parsed on every frame -- a paragraph
 * becomes a list, an open fence closes.
 *
 * Fences are counted first: inside an open one the whole block is provisional,
 * including any blank lines in it, so the boundary is that fence's own start.
 */
export const lastBlockStart = (src: string): number => {
  const fences = src.split("```").length - 1
  if (fences % 2 === 1) return src.lastIndexOf("```")
  const closed = src.lastIndexOf("```")
  const gap = src.lastIndexOf("\n\n")
  // A closed fence ends a block as firmly as a blank line does.
  if (closed !== -1 && closed + 3 > gap) return closed + 3
  return gap === -1 ? 0 : gap + 2
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
  const answerTail = document.createElement("div")
  answerTail.className = "tail"
  answerBody.appendChild(answerTail)
  const note = document.createElement("div")
  note.className = "note"
  note.hidden = true

  root.append(fold, answerBody, note)
  into.appendChild(root)
  return { root, reasoning: "", answer: "", settled: 0, fold, reasoningBody,
           answerBody, answerTail, note }
}

/** Render what has arrived, re-parsing only the block still being written.
 *
 * The whole answer used to be re-parsed and every node replaced on each frame,
 * which is O(reply^2) over a stream and throws away the DOM under the reader's
 * selection. Finished blocks are appended once and never touched again; the tail
 * is the only node a frame replaces.
 */
export const paint = (t: Turn): void => {
  if (t.reasoning !== "") {
    t.fold.hidden = false
    t.reasoningBody.replaceChildren(document.createTextNode(t.reasoning))
  }
  const cut = lastBlockStart(t.answer)
  if (cut > t.settled) {
    // insertBefore, not append: the tail has to stay last.
    t.answerBody.insertBefore(markdown(t.answer.slice(t.settled, cut)), t.answerTail)
    t.settled = cut
  }
  t.answerTail.replaceChildren(markdown(t.answer.slice(t.settled)))
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
