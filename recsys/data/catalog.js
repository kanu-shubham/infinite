// Video catalogue + tag vocabulary.
//
// Each video carries a tag-vector that doubles as its content embedding.
// In production these embeddings would come from a learned model trained on
// titles/descriptions/co-watch signals. The shape of the rest of the pipeline
// doesn't change.

const TAG_VOCAB = [
  "tech", "ai", "gaming", "music", "comedy",
  "news", "sports", "cooking", "travel", "fitness",
  "education", "science", "diy", "fashion", "movies",
];

const CHANNELS = [
  { id: "ch_pixelpilot",  name: "PixelPilot",    primary: ["tech", "ai"] },
  { id: "ch_byteforge",   name: "ByteForge",     primary: ["tech", "education", "science"] },
  { id: "ch_loopstation", name: "LoopStation",   primary: ["music"] },
  { id: "ch_punchline",   name: "Punchline",     primary: ["comedy"] },
  { id: "ch_dailyfeed",   name: "DailyFeed",     primary: ["news"] },
  { id: "ch_arenacast",   name: "ArenaCast",     primary: ["sports", "fitness"] },
  { id: "ch_skillet",     name: "Skillet",       primary: ["cooking"] },
  { id: "ch_wanderlist",  name: "Wanderlist",    primary: ["travel"] },
  { id: "ch_pulse",       name: "Pulse Trainer", primary: ["fitness"] },
  { id: "ch_chalkboard",  name: "Chalkboard",    primary: ["education", "science"] },
  { id: "ch_makerlab",    name: "MakerLab",      primary: ["diy", "tech"] },
  { id: "ch_runway",      name: "Runway",        primary: ["fashion"] },
  { id: "ch_reelroom",    name: "ReelRoom",      primary: ["movies"] },
  { id: "ch_neuralnotes", name: "Neural Notes",  primary: ["ai", "science"] },
  { id: "ch_couchgaming", name: "CouchGaming",   primary: ["gaming"] },
];

const TITLE_TEMPLATES = {
  tech:      ["Why {x} Just Broke the Internet", "Building a {x} from Scratch"],
  ai:        ["This AI Can {x}", "Inside the New {x} Model"],
  gaming:    ["Speedrunning {x} in Under an Hour", "Ranking Every {x} Mechanic"],
  music:     ["{x} - Live Session", "Lo-fi Beats for {x}"],
  comedy:    ["What if {x} Was a Sitcom?", "{x} but Make it Awkward"],
  news:      ["Briefing: {x} Explained", "Why {x} Matters Today"],
  sports:    ["Top 10 {x} Plays of the Week", "Tactics Breakdown: {x}"],
  cooking:   ["15-Minute {x}", "Pro Chef Reacts to {x}"],
  travel:    ["48 Hours in {x}", "Hidden Spots in {x}"],
  fitness:   ["Full-Body {x} Routine", "30-Day {x} Challenge"],
  education: ["The History of {x}, Briefly", "{x} - From First Principles"],
  science:   ["The Physics of {x}", "Why {x} Surprises Researchers"],
  diy:       ["Building My Own {x}", "{x} - Weekend Project"],
  fashion:   ["Styling {x} for Spring", "Inside the {x} Studio"],
  movies:    ["Why {x} Still Holds Up", "Hidden Details in {x}"],
};

const TOPIC_WORDS = {
  tech:      ["the M5 chip", "RISC-V", "passwordless login", "a 1KB OS"],
  ai:        ["draw", "compose music", "code review", "play chess"],
  gaming:    ["Hollow Knight", "Elden Ring", "Tetris", "Celeste"],
  music:     ["a rainy commute", "late nights", "deep focus", "Sunday morning"],
  comedy:    ["airport queues", "tax season", "group projects", "open offices"],
  news:      ["the energy bill", "the climate report", "the new tariff", "AI policy"],
  sports:    ["the Premier League", "F1 Monaco", "the NBA finals", "Wimbledon"],
  cooking:   ["ramen", "carbonara", "sourdough", "miso glaze"],
  travel:    ["Lisbon", "Kyoto", "Reykjavik", "Cape Town"],
  fitness:   ["mobility", "kettlebell", "post-run", "core"],
  education: ["calculus", "the printing press", "double-entry bookkeeping", "tectonics"],
  science:   ["a black hole", "mRNA", "neutrinos", "the Coriolis effect"],
  diy:       ["a standing desk", "a smart mirror", "a bike rack", "a soldering station"],
  fashion:   ["denim", "linen", "a capsule wardrobe", "vintage outerwear"],
  movies:    ["Heat", "Akira", "Spirited Away", "Blade Runner"],
};

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

function buildTagVector(tags) {
  const v = new Array(TAG_VOCAB.length).fill(0);
  tags.forEach((t, i) => {
    const idx = TAG_VOCAB.indexOf(t);
    if (idx >= 0) v[idx] = i === 0 ? 1.0 : 0.6;
  });
  return v;
}

function generateCatalog(count = 240, seed = 42) {
  const rng = mulberry32(seed);
  const pick = (arr) => arr[Math.floor(rng() * arr.length)];
  const videos = [];
  const now = Date.now();

  for (let i = 0; i < count; i++) {
    const channel = pick(CHANNELS);
    const primaryTag = pick(channel.primary);
    const extraTag = rng() < 0.45
      ? pick(TAG_VOCAB.filter((t) => !channel.primary.includes(t)))
      : null;
    const tags = extraTag ? [primaryTag, extraTag] : [primaryTag];

    const titleTpl = pick(TITLE_TEMPLATES[primaryTag]);
    const topic = pick(TOPIC_WORDS[primaryTag]);
    const title = titleTpl.replace("{x}", topic);

    const ageDays = Math.floor(-Math.log(1 - rng()) * 7);
    const uploadedAt = now - ageDays * 24 * 60 * 60 * 1000;

    const base = Math.exp(rng() * 4);
    const popularity = base * (1 + (channel.primary.includes(primaryTag) ? 0.5 : 0));

    const durationSec = 60 + Math.floor(rng() * (primaryTag === "music" ? 600 : 1200));

    videos.push({
      id: `vid_${i.toString(36)}`,
      title,
      channelId: channel.id,
      channelName: channel.name,
      tags,
      tagVector: buildTagVector(tags),
      durationSec,
      uploadedAt,
      popularity,
    });
  }
  return videos;
}

class Catalog {
  constructor(videos) {
    this.videos = videos;
    this.byId = new Map(videos.map((v) => [v.id, v]));
  }
  all() { return this.videos; }
  get(id) { return this.byId.get(id); }
  size() { return this.videos.length; }
}

module.exports = { TAG_VOCAB, CHANNELS, Catalog, generateCatalog };
