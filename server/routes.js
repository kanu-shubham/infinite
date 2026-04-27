"use strict";

const http = require("node:http");
const { URL } = require("node:url");

function send(res, status, body) {
  const payload = typeof body === "string" ? body : JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  });
  res.end(payload);
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1e6) {
        req.destroy();
        reject(new Error("payload too large"));
      }
    });
    req.on("end", () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function makeRoutes(service) {
  return [
    {
      method: "GET",
      pattern: /^\/api\/pricing\/health$/,
      handler: async () => ({ ok: true, ts: Date.now() }),
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/hotels$/,
      handler: async () => ({ hotels: service.repo.listHotels() }),
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/quote$/,
      handler: async (req) => {
        const body = await readJson(req);
        if (body.hotelId == null) throw httpError(400, "hotelId required");
        return service.quote({
          hotelId: Number(body.hotelId),
          context: body.context || {},
          userId: body.userId || "anon",
          objective: body.objective,
        });
      },
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/feedback$/,
      handler: async (req) => {
        const body = await readJson(req);
        const required = ["hotelId", "price"];
        for (const k of required) {
          if (body[k] == null) throw httpError(400, `${k} required`);
        }
        return service.recordFeedback({
          decisionId: body.decisionId,
          hotelId: Number(body.hotelId),
          price: Number(body.price),
          units: Number(body.units != null ? body.units : body.booked ? 1 : 0),
          booked: !!body.booked,
          context: body.context,
        });
      },
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/train$/,
      handler: async () => service.trainAll(),
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/explain\/(\d+)$/,
      handler: async (req, _res, match, url) => {
        const hotelId = Number(match[1]);
        const context = {
          checkIn: url.searchParams.get("checkIn") || undefined,
          nights: url.searchParams.get("nights")
            ? Number(url.searchParams.get("nights"))
            : undefined,
          leadTimeDays: url.searchParams.get("leadTimeDays")
            ? Number(url.searchParams.get("leadTimeDays"))
            : undefined,
        };
        return service.explain({ hotelId, context });
      },
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/elasticity\/(\d+)$/,
      handler: async (_req, _res, match) => {
        const hotelId = Number(match[1]);
        const fit = service.elasticityFits.get(hotelId);
        if (!fit) throw httpError(404, "no elasticity fit yet — train first");
        return { hotelId, ...fit };
      },
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/audit$/,
      handler: async (_req, _res, _m, url) => {
        const limit = Number(url.searchParams.get("limit") || 50);
        return { entries: service.repo.recentAudit(limit) };
      },
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/evaluate$/,
      handler: async (req) => {
        const body = await readJson(req);
        return service.evaluatePolicy({ uplift: Number(body.uplift) || 0 });
      },
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/drift$/,
      handler: async () => service.driftReport(),
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/experiments$/,
      handler: async (req) => {
        const body = await readJson(req);
        if (!body.id || !Array.isArray(body.variants)) {
          throw httpError(400, "id and variants required");
        }
        service.repo.setExperiment(body.id, {
          variants: body.variants,
          status: "active",
        });
        return { ok: true };
      },
    },
    {
      method: "GET",
      pattern: /^\/api\/pricing\/experiments$/,
      handler: async () => ({ experiments: service.repo.listExperiments() }),
    },
    {
      method: "POST",
      pattern: /^\/api\/pricing\/experiments\/assign$/,
      handler: async (req) => {
        const body = await readJson(req);
        if (!body.experimentId || !body.unitId) {
          throw httpError(400, "experimentId and unitId required");
        }
        const variant = service.assignExperiment(body.experimentId, body.unitId);
        if (!variant) throw httpError(404, "experiment not found");
        return { experimentId: body.experimentId, unitId: body.unitId, variant };
      },
    },
  ];
}

function httpError(status, message) {
  const err = new Error(message);
  err.status = status;
  return err;
}

function createServer(service) {
  const routes = makeRoutes(service);
  return http.createServer(async (req, res) => {
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      });
      return res.end();
    }
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    for (const route of routes) {
      if (route.method !== req.method) continue;
      const match = route.pattern.exec(url.pathname);
      if (!match) continue;
      try {
        const body = await route.handler(req, res, match, url);
        return send(res, 200, body);
      } catch (err) {
        return send(res, err.status || 500, { error: err.message });
      }
    }
    return send(res, 404, { error: "not_found", path: url.pathname });
  });
}

module.exports = { createServer, makeRoutes };
