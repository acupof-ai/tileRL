"""The landing page: what tileRL is, the target matrix, and entry to the playground.

Kept as a Python string rather than package data so `tests/test_chat_ui.py` can
import and parse it, which is how the XSS and CSS-scoping gates work. The chat
playground used to live here too; it is now a built bundle under `static/`, from
the TypeScript sources in `web/`.
"""

_LANDING = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tilerl</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1115; --panel: #171a21; --panel2: #1e222b; --line: #2a2f3a;
    --fg: #e6e8eb; --dim: #8b93a1; --accent: #7aa2f7; --accent2: #9ece6a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh;
    font: 15px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    background: radial-gradient(1200px 600px at 50% -10%, #1a1f2b 0%, var(--bg) 60%);
    color: var(--fg); display: flex; flex-direction: column; align-items: center;
  }
  main { width: 100%; max-width: 860px; padding: 64px 24px 80px; }
  .brand { font-size: 44px; font-weight: 800; letter-spacing: -.5px; color: var(--accent); }
  .tag { font-size: 19px; color: var(--fg); margin: 10px 0 4px; }
  .sub { color: var(--dim); font-size: 15px; max-width: 640px; }
  .cta { margin: 34px 0 48px; display: flex; gap: 12px; flex-wrap: wrap; }
  .cta a { text-decoration: none; padding: 12px 26px; border-radius: 10px; font-weight: 600; }
  .cta .primary { background: var(--accent); color: #0f1115; }
  .cta .ghost { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); }
  h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .6px; color: var(--dim);
       margin: 40px 0 14px; font-weight: 700; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .card b { color: var(--accent2); }
  .card p { color: var(--dim); font-size: 14px; margin: 6px 0 0; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); }
  th { color: var(--dim); font-weight: 600; font-size: 12px; text-transform: uppercase; }
  td b { color: var(--accent2); font-variant-numeric: tabular-nums; }
  .ok { color: var(--accent2); } .wip { color: #e0af68; }
  code { background: var(--panel2); padding: 2px 7px; border-radius: 5px; font-size: 13px; }
  footer { color: var(--dim); font-size: 13px; margin-top: 44px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<main>
  <div class="brand">tilerl</div>
  <div class="tag">Cross-platform train + inference for <b>Qwen3.8-27B (NVFP4)</b>, one TileLang kernel source.</div>
  <div class="sub">One kernel tree compiles for CPU, Metal, and CUDA — including <b>Volta / sm70</b>,
    the first pre-Ampere card to run the stack. Paged KV with a prefix cache, an on-policy-distillation
    trainer that shares the serving engine — no second stack.</div>

  <div class="cta">
    <a class="primary" href="/chat">Open the playground &rarr;</a>
    <a class="ghost" href="/v1/models">API: /v1/models</a>
    <a class="ghost" href="/health">Health</a>
  </div>

  <h2>Serving this model</h2>
  <div class="grid">
    <div class="card"><b id="model">…</b><p id="mstat">connecting…</p></div>
    <div class="card"><b>NVFP4 W4A16</b><p>4-bit weights in HBM, dequant in-kernel, fp16/f32 compute</p></div>
    <div class="card"><b>Paged KV + prefix cache</b><p>a repeated prefix is retained in HBM and skips prefill</p></div>
  </div>

  <h2>Target matrix</h2>
  <table>
    <tr><th>Target</th><th>Status</th><th>Decode B=1</th></tr>
    <tr><td>CUDA sm90 (H20)</td><td class="ok">shipped</td><td><b>92.4</b> tok/s</td></tr>
    <tr><td>CUDA sm70 (V100)</td><td class="ok">this build</td><td><b>19.9</b> tok/s <span class="wip">(GEMV opt in progress)</span></td></tr>
    <tr><td>CPU</td><td class="ok">CI / dev path</td><td>—</td></tr>
    <tr><td>Metal</td><td class="ok">local</td><td>—</td></tr>
  </table>

  <h2>Two ways in</h2>
  <div class="grid">
    <div class="card"><b>Chat</b><p>Stream from the model directly. Live TTFT and tok/s. <a href="/chat">Chat &rarr;</a></p></div>
  </div>

  <footer>OpenAI-compatible at <code>POST /v1/chat/completions</code></footer>
</main>
<script>
  fetch("/v1/models").then(r => r.json()).then(j => {
    document.getElementById("model").textContent = j.data[0].id;
    document.getElementById("mstat").textContent = "ready";
  }).catch(() => { document.getElementById("mstat").textContent = "offline"; });
</script>
</body>
</html>"""

