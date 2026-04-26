// HTTP routing for the UI-facing API.
//
// Each handler returns a plain { status, body } so the same routing table can
// be reused under any transport (raw http, Express, serverless). Keeping the
// handlers transport-agnostic also makes them trivially testable.

function ok(body)              { return { status: 200, body }; }
function created(body)         { return { status: 201, body }; }
function badRequest(message)   { return { status: 400, body: { error: message } }; }
function notFound()            { return { status: 404, body: { error: "not found" } }; }

// GET /v1/recommendations?userId=<id>&page=<n>&pageSize=<n>
function getRecommendations(recommender, query) {
  const userId = query.userId;
  if (!userId) return badRequest("userId is required");

  const page = clampInt(query.page, 1, 1, 1000);
  const pageSize = clampInt(query.pageSize, 8, 1, 50);

  const blocklist = (query.blocklist || "").split(",").filter(Boolean);

  const result = recommender.recommend(userId, { page, pageSize, blocklist });
  return ok(result);
}

// POST /v1/events    body: { userId, videoId, type, watchRatio? }
function postEvent(recommender, body) {
  if (!body || typeof body !== "object") return badRequest("body must be JSON object");
  const { userId, videoId, type } = body;
  if (!userId || !videoId || !type) {
    return badRequest("userId, videoId, type are required");
  }
  if (!["watch", "like", "skip"].includes(type)) {
    return badRequest(`type must be watch|like|skip (got ${type})`);
  }
  if (type === "watch") {
    const r = body.watchRatio;
    if (r != null && (typeof r !== "number" || r < 0 || r > 1)) {
      return badRequest("watchRatio must be a number in [0,1]");
    }
  }
  try {
    const stored = recommender.ingestEvent({
      userId,
      videoId,
      type,
      watchRatio: body.watchRatio,
    });
    return created({ accepted: true, eventTs: stored.ts });
  } catch (e) {
    return badRequest(e.message);
  }
}

// GET /v1/profile?userId=<id>
function getProfile(recommender, query) {
  const userId = query.userId;
  if (!userId) return badRequest("userId is required");
  const profile = recommender.profileStore.get(userId);
  // Strip the dense vector from the public surface; expose top tags instead.
  const { topTags } = require("../model/userProfile");
  return ok({
    userId,
    eventCount: profile.eventCount,
    watchHistory: profile.watchHistory,
    liked: profile.liked,
    skipped: profile.skipped,
    topTags: topTags(profile, 8),
    channelAffinity: profile.channelAffinity,
    updatedAt: profile.updatedAt,
  });
}

// DELETE /v1/profile?userId=<id>
function deleteProfile(recommender, query) {
  const userId = query.userId;
  if (!userId) return badRequest("userId is required");
  recommender.profileStore.delete(userId);
  return ok({ deleted: true });
}

function clampInt(raw, def, min, max) {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return def;
  return Math.min(max, Math.max(min, n));
}

// Routing table: (method, pathPrefix) -> handler signature.
// The transport adapter parses `query` and `body` then dispatches here.
function buildRoutes(recommender) {
  return [
    { method: "GET",    path: "/v1/recommendations", handler: (q, b) => getRecommendations(recommender, q) },
    { method: "POST",   path: "/v1/events",          handler: (q, b) => postEvent(recommender, b) },
    { method: "GET",    path: "/v1/profile",         handler: (q, b) => getProfile(recommender, q) },
    { method: "DELETE", path: "/v1/profile",         handler: (q, b) => deleteProfile(recommender, q) },
    { method: "GET",    path: "/health",             handler: () => ok({ ok: true }) },
  ];
}

module.exports = { buildRoutes, notFound };
