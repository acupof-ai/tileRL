import { defineConfig } from "vite"

// The bundle is committed under src/tilerl/static/ and served by FastAPI, so the
// Python side never needs node. Relative base: the page is mounted at / and at
// /chat, and an absolute /assets path 404s under any future prefix mount.
export default defineConfig({
  base: "./",
  build: {
    outDir: "../src/tilerl/static",
    emptyOutDir: true,
    target: "es2022",
    // One chunk, so there is nothing to preload; the polyfill is 800 bytes of
    // MutationObserver that only exists to warm chunks a split build would emit.
    modulePreload: false,
  },
})
