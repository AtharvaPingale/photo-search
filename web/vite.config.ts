import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// The API serves the built app from web/dist; in dev, Vite proxies /api to it so
// the auth cookie stays same-origin.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icons/apple-touch-icon.png", "icons/favicon.svg"],
      manifest: {
        name: "Photo Search",
        short_name: "Photos",
        description: "Search your photo library in plain language.",
        start_url: "/",
        scope: "/",
        display: "standalone",
        orientation: "portrait",
        background_color: "#0f1115",
        theme_color: "#0f1115",
        icons: [
          { src: "icons/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icons/icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icons/maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/api\//, /^\/docs/],
        runtimeCaching: [
          {
            // thumbnail URLs carry the content hash, so cache-first is always correct
            urlPattern: ({ url }) => url.pathname.startsWith("/api/photos/") && url.pathname.endsWith("/thumb"),
            handler: "CacheFirst",
            options: {
              cacheName: "thumbs",
              expiration: { maxEntries: 5000, maxAgeSeconds: 60 * 60 * 24 * 60 },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith("/api/photos/") && url.pathname.endsWith("/display"),
            handler: "CacheFirst",
            options: {
              cacheName: "display",
              expiration: { maxEntries: 300, maxAgeSeconds: 60 * 60 * 24 * 14 },
              cacheableResponse: { statuses: [200] },
            },
          },
        ],
      },
    }),
  ],
  server: {
    host: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: { target: "es2020", sourcemap: false },
});
