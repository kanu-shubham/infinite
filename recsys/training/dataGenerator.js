// Synthetic interaction data generator.
//
// Produces a labeled dataset of (userId, videoId, label, watchRatio) tuples
// that we can train MF + ranker on. We invent N users, give each a small set
// of latent tag preferences, then sample interactions whose label is
// stochastic but biased by tag overlap. This is the closest we can get to
// "real engagement logs" inside a self-contained repo.
//
// In production this generator is replaced by the event-log → ETL pipeline.

const { TAG_VOCAB } = require("../data/catalog");

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }

// Build N synthetic users. Each gets 2–4 preferred tags + a "channel
// loyalty" weight drawn from a Beta-ish distribution so some users stick to
// channels and others don't.
function generateUsers(rng, n) {
  const users = [];
  for (let i = 0; i < n; i++) {
    const k = 2 + Math.floor(rng() * 3);
    const prefs = new Set();
    while (prefs.size < k) prefs.add(sample(rng, TAG_VOCAB));
    users.push({
      userId: `u_${i.toString(36)}`,
      preferredTags: [...prefs],
      channelLoyalty: 0.2 + rng() * 0.6,        // [0.2, 0.8]
      noise: 0.02 + rng() * 0.05,              // label flip rate (kept low so signal dominates)
    });
  }
  return users;
}

// Probability of a "satisfied watch" given a user/video pair. Mixes:
//   - tag overlap
//   - per-channel loyalty (we pin one liked channel per user from their first prefs)
//   - small popularity bias (busy creators get watched more)
function satisfactionProb(user, video, loyalChannelId) {
  const overlap = video.tags.reduce(
    (acc, t) => acc + (user.preferredTags.includes(t) ? 1 : 0),
    0,
  ) / Math.max(1, video.tags.length);

  const channelBoost = video.channelId === loyalChannelId ? user.channelLoyalty : 0;
  const popularityBoost = Math.log1p(video.popularity) / 50;

  let p = 0.05 + 0.7 * overlap + 0.3 * channelBoost + popularityBoost;
  return Math.max(0.01, Math.min(0.95, p));
}

function generateInteractions(rng, users, catalog, perUser = 30) {
  const videos = catalog.all();
  const events = [];
  for (const user of users) {
    // Pick a "loyal channel" from a video matching the user's top tag.
    const seed = videos.find((v) => v.tags.includes(user.preferredTags[0]));
    const loyalChannelId = seed ? seed.channelId : null;

    // Sample candidate videos. Mix of tag-matching (~60%) and random (~40%)
    // to simulate exposure — the user sees both relevant + irrelevant items
    // in real life, but a recommender has surfaced more of the former.
    const tagMatching = videos.filter((v) => v.tags.some((t) => user.preferredTags.includes(t)));
    const exposed = new Set();
    while (exposed.size < perUser) {
      const v = rng() < 0.6 && tagMatching.length > 0
        ? sample(rng, tagMatching)
        : sample(rng, videos);
      exposed.add(v.id);
    }

    for (const vid of exposed) {
      const v = catalog.get(vid);
      let p = satisfactionProb(user, v, loyalChannelId);
      // Apply per-user noise.
      if (rng() < user.noise) p = 1 - p;

      const u = rng();
      let label, type, watchRatio;
      if (u < p) {
        label = 1;
        type = "watch";
        watchRatio = 0.6 + rng() * 0.4; // 0.6..1.0
      } else if (u < p + 0.15) {
        // Started but didn't finish — soft positive.
        label = 0;
        type = "watch";
        watchRatio = rng() * 0.4; // 0..0.4
      } else {
        label = 0;
        type = "skip";
        watchRatio = 0;
      }
      events.push({
        userId: user.userId,
        videoId: v.id,
        type,
        watchRatio,
        label,
      });
    }
  }
  return events;
}

// 80/20 split per user, chronological-ish but here we just shuffle since
// events are unordered.
function trainTestSplit(rng, events, testFrac = 0.2) {
  const byUser = new Map();
  for (const e of events) {
    if (!byUser.has(e.userId)) byUser.set(e.userId, []);
    byUser.get(e.userId).push(e);
  }
  const train = [];
  const test = [];
  for (const [, list] of byUser) {
    // Fisher-Yates shuffle.
    for (let i = list.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [list[i], list[j]] = [list[j], list[i]];
    }
    const cut = Math.floor(list.length * (1 - testFrac));
    train.push(...list.slice(0, cut));
    test.push(...list.slice(cut));
  }
  return { train, test };
}

function generate(catalog, { numUsers = 200, perUser = 30, seed = 7 } = {}) {
  const rng = mulberry32(seed);
  const users = generateUsers(rng, numUsers);
  const events = generateInteractions(rng, users, catalog, perUser);
  const { train, test } = trainTestSplit(rng, events, 0.2);
  return { users, events, train, test };
}

module.exports = { generate, mulberry32 };
