// Sample API client — what the UI would import.
//
// Plain fetch wrapper, no React. The UI calls `getRecommendations` once on
// page load and on each "load more" intersection; it calls `postEvent`
// fire-and-forget on every watch / like / skip; and may call `getProfile` to
// render a "your interests" panel.
//
// Works in browsers natively. In Node 18+ `fetch` is also global.

class RecsClient {
  constructor({ baseUrl = "http://localhost:8787", userId, fetchImpl } = {}) {
    if (!userId) throw new Error("userId is required");
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.userId = userId;
    this.fetch = fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
    if (!this.fetch) throw new Error("no fetch implementation available");
  }

  async getRecommendations({ page = 1, pageSize = 8 } = {}) {
    const u = new URL(this.baseUrl + "/v1/recommendations");
    u.searchParams.set("userId", this.userId);
    u.searchParams.set("page", String(page));
    u.searchParams.set("pageSize", String(pageSize));
    const r = await this.fetch(u.toString());
    if (!r.ok) throw new Error(`recs ${r.status}`);
    return r.json();
  }

  // Fire-and-forget telemetry. Failures are swallowed so a bad network never
  // blocks the player.
  postEvent(event) {
    return this.fetch(this.baseUrl + "/v1/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: this.userId, ...event }),
      keepalive: true,
    }).catch(() => {});
  }

  watch(videoId, watchRatio)  { return this.postEvent({ videoId, type: "watch", watchRatio }); }
  like(videoId)               { return this.postEvent({ videoId, type: "like" }); }
  skip(videoId)               { return this.postEvent({ videoId, type: "skip" }); }

  async getProfile() {
    const u = new URL(this.baseUrl + "/v1/profile");
    u.searchParams.set("userId", this.userId);
    const r = await this.fetch(u.toString());
    if (!r.ok) throw new Error(`profile ${r.status}`);
    return r.json();
  }

  async resetProfile() {
    const u = new URL(this.baseUrl + "/v1/profile");
    u.searchParams.set("userId", this.userId);
    const r = await this.fetch(u.toString(), { method: "DELETE" });
    if (!r.ok) throw new Error(`reset ${r.status}`);
    return r.json();
  }
}

module.exports = { RecsClient };
