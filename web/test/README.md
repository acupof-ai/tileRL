# Node client tests

Run with `pnpm test` (from `web/`). Node built-in test runner (`node:test`),
zero dependencies, type-stripping the TypeScript sources directly.

Local-only until CI grows a Node step; the committed-artifact guards stay in
`tests/test_chat_ui.py` (pytest), which CI does run.
