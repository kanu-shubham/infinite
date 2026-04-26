// User profile model.
//
// Pure functions over an immutable profile shape. The service layer is
// responsible for persistence — this file only knows how to *build* and
// *update* a profile from events.

const { TAG_VOCAB } = require("../data/catalog");

const EVENT_WEIGHTS = {
  watch: 1.0,    // additionally scaled by watchRatio in [0,1]
  like:  0.8,
  skip: -0.6,
};

const DECAY_PER_UPDATE = 0.985;

function emptyProfile(userId) {
  return {
    userId,
    tagAffinity: new Array(TAG_VOCAB.length).fill(0),
    channelAffinity: {},      // channelId -> score
    watchHistory: [],         // most-recent-last, capped
    liked: [],
    skipped: [],
    eventCount: 0,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

// Returns a new profile with the event applied. Pure — does not mutate input.
function applyEvent(profile, event, catalog) {
  const video = catalog.get(event.videoId);
  if (!video) return profile;

  const next = {
    ...profile,
    tagAffinity: profile.tagAffinity.slice(),
    channelAffinity: { ...profile.channelAffinity },
    watchHistory: profile.watchHistory.slice(),
    liked: profile.liked.slice(),
    skipped: profile.skipped.slice(),
    eventCount: profile.eventCount + 1,
    updatedAt: Date.now(),
  };

  // Apply gentle decay so old interests fade.
  for (let i = 0; i < next.tagAffinity.length; i++) {
    next.tagAffinity[i] *= DECAY_PER_UPDATE;
  }
  for (const ch of Object.keys(next.channelAffinity)) {
    next.channelAffinity[ch] *= DECAY_PER_UPDATE;
  }

  const baseWeight = EVENT_WEIGHTS[event.type] ?? 0;
  const weight = event.type === "watch"
    ? baseWeight * Math.max(0, Math.min(1, event.watchRatio ?? 0))
    : baseWeight;

  for (let i = 0; i < video.tagVector.length; i++) {
    next.tagAffinity[i] += weight * video.tagVector[i];
    if (next.tagAffinity[i] < 0) next.tagAffinity[i] = 0;
  }

  next.channelAffinity[video.channelId] =
    (next.channelAffinity[video.channelId] || 0) + weight * 0.5;
  if (next.channelAffinity[video.channelId] < 0) {
    delete next.channelAffinity[video.channelId];
  }

  if (event.type === "watch") {
    next.watchHistory.push(video.id);
    if (next.watchHistory.length > 50) next.watchHistory.shift();
  } else if (event.type === "like" && !next.liked.includes(video.id)) {
    next.liked.push(video.id);
  } else if (event.type === "skip" && !next.skipped.includes(video.id)) {
    next.skipped.push(video.id);
  }

  return next;
}

function topTags(profile, n = 5) {
  return profile.tagAffinity
    .map((v, i) => ({ tag: TAG_VOCAB[i], v }))
    .filter((t) => t.v > 0.01)
    .sort((a, b) => b.v - a.v)
    .slice(0, n);
}

function isColdStart(profile) {
  return profile.watchHistory.length === 0 && profile.liked.length === 0;
}

module.exports = {
  emptyProfile,
  applyEvent,
  topTags,
  isColdStart,
  EVENT_WEIGHTS,
};
