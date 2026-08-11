const { createProxyMiddleware } = require("http-proxy-middleware");

/**
 * Forwards the ML pipeline API to the FastAPI backend during development, so
 * the frontend can call same-origin `/api/...` paths and CORS stays out of the
 * local loop.
 *
 * This is deliberately a setupProxy file rather than the `proxy` field in
 * package.json: the shorthand turns on webpack-dev-server's host check, which
 * fails to start in containers and VMs that have no resolvable LAN address.
 *
 * Point ML_API_TARGET elsewhere if the backend is not on port 8000.
 */
module.exports = function setupProxy(app) {
  app.use(
    "/api",
    createProxyMiddleware({
      target: process.env.ML_API_TARGET || "http://localhost:8000",
      changeOrigin: true,
    })
  );
};
