import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiUrl = env.VITE_API_URL || "http://127.0.0.1:8000";

  const proxyConfig = {
    "/api": {
      target: apiUrl,
      changeOrigin: true,
      secure: false,
      ws: false,
      timeout: 120_000,
      proxyTimeout: 120_000,
      headers: {
        // Helps SSE + long-lived connections
        Connection: "keep-alive",
        Accept: "text/event-stream",
      },
      configure: (proxy: any) => {
        proxy.on("proxyReq", (proxyReq: any, req: any) => {
          // Ensure SSE requests express intent
          if (req.url?.includes("/api/query")) {
            proxyReq.setHeader("Accept", "text/event-stream");
            proxyReq.setHeader("Cache-Control", "no-cache");
            proxyReq.setHeader("Connection", "keep-alive");
          }
        });

        proxy.on("proxyRes", (proxyRes: any, req: any) => {
          if (req.url?.includes("/api/query")) {
            // Nginx-specific but harmless elsewhere
            proxyRes.headers["x-accel-buffering"] = "no";
            // Avoid caches messing with streaming
            proxyRes.headers["cache-control"] = "no-cache";
          }
        });
      },
    },
    "/health": {
      target: apiUrl,
      changeOrigin: true,
      secure: false,
    },
  };

  return {
    plugins: [react()],
    server: {
      proxy: proxyConfig,
    },
  };
});
