// Minimal Node http server that adapts the routing table to HTTP.
//
// Run:    node recsys/api/server.js
// Env:    PORT (default 8787), CATALOG_SIZE (default 240)
//
// The UI calls these endpoints. There are no UI-side stubs in this repo;
// the contract is JSON-over-HTTP defined in api/routes.js.

const http = require("http");
const url = require("url");
const { generateCatalog, Catalog } = require("../data/catalog");
const { Recommender } = require("../service/recommender");
const { InMemoryProfileStore } = require("../service/profileStore");
const { InMemoryEventLog } = require("../service/eventLog");
const { LearnedModel } = require("../model/learnedModel");
const { buildRoutes, notFound } = require("./routes");

function readJsonBody(req, max = 1 << 20) {
  return new Promise((resolve, reject) => {
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
      if (raw.length > max) {
        reject(new Error("payload too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!raw) return resolve(null);
      try { resolve(JSON.parse(raw)); }
      catch (e) { reject(new Error("invalid JSON")); }
    });
    req.on("error", reject);
  });
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function send(res, { status, body }) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}

function buildServer({ catalogSize = 240, weights, useLearned = true } = {}) {
  const catalog = new Catalog(generateCatalog(catalogSize));
  const profileStore = new InMemoryProfileStore();
  const eventLog = new InMemoryEventLog();
  const learnedModel = useLearned ? LearnedModel.loadFromDisk() : null;
  if (useLearned) {
    if (learnedModel) {
      console.log(`[server] loaded learned model (auc=${learnedModel.meta?.metrics?.auc?.toFixed(3)} recall@10=${learnedModel.meta?.metrics?.recallAt10?.toFixed(3)})`);
    } else {
      console.log("[server] no artifacts found in recsys/artifacts/ — falling back to heuristic model. Run `node recsys/training/runTrain.js` to train.");
    }
  }
  const recommender = new Recommender({ catalog, profileStore, eventLog, weights, learnedModel });
  const routes = buildRoutes(recommender);

  const server = http.createServer(async (req, res) => {
    setCors(res);
    if (req.method === "OPTIONS") { res.statusCode = 204; res.end(); return; }

    const parsed = url.parse(req.url, true);
    const route = routes.find((r) => r.method === req.method && r.path === parsed.pathname);
    if (!route) { send(res, notFound()); return; }

    let body = null;
    if (req.method === "POST" || req.method === "PUT") {
      try { body = await readJsonBody(req); }
      catch (e) { send(res, { status: 400, body: { error: e.message } }); return; }
    }

    try {
      const result = route.handler(parsed.query, body);
      send(res, result);
    } catch (e) {
      send(res, { status: 500, body: { error: e.message } });
    }
  });

  // Expose the recommender for tests / demos that import this module.
  server.recommender = recommender;
  return server;
}

if (require.main === module) {
  const port = parseInt(process.env.PORT, 10) || 8787;
  const catalogSize = parseInt(process.env.CATALOG_SIZE, 10) || 240;
  const server = buildServer({ catalogSize });
  server.listen(port, () => {
    console.log(`recsys api listening on http://localhost:${port}`);
    console.log(`  GET  /v1/recommendations?userId=<id>&page=1&pageSize=8`);
    console.log(`  POST /v1/events     { userId, videoId, type, watchRatio? }`);
    console.log(`  GET  /v1/profile?userId=<id>`);
  });
}

module.exports = { buildServer };
