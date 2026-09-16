import { defineConfig } from "vite"

// The bundle is committed under src/tilerl/static/ and served by FastAPI, so the
// Python side never needs node. Relative base: the page is mounted at / and at
// /chat, and an absolute /assets path 404s under any future prefix mount.
//
// Dev-only proxy against a live server over an ssh tunnel
// (`ssh -L 18000:localhost:8000 v100`, then `pnpm dev`): the page derives its
// socket from window.location, so without this the WS/HTTP calls would hit
// vite's port with no backend. Not part of the production build.
export default defineConfig({
  base: "./",
  server: {
    proxy: {
      "/ws": { target: "http://localhost:18000", ws: true },
      "/health": "http://localhost:18000",
      "/v1": "http://localhost:18000",
    },
  },
  build: {
    outDir: "../src/tilerl/static",
    emptyOutDir: true,
    target: "es2022",
    // One chunk, so there is nothing to preload; the polyfill is 800 bytes of
    // MutationObserver that only exists to warm chunks a split build would emit.
    modulePreload: false,
  },
})
