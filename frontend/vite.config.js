import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Served by FastAPI at /ui/, so assets must resolve relative to that base.
export default defineConfig({
  base: "/ui/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/interview-types": "http://127.0.0.1:8000",
      "/interviews": "http://127.0.0.1:8000",
      "/candidate-contexts": "http://127.0.0.1:8000",
      "/candidates": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
