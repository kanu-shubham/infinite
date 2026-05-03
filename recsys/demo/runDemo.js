// End-to-end driver. Boots the recommender in-process (no HTTP), simulates a
// new user who develops a clear taste over a few sessions, and prints the
// feed before/after to show the system adapting.
//
// Run: node recsys/demo/runDemo.js

const { generateCatalog, Catalog } = require("../data/catalog");
const { Recommender } = require("../service/recommender");
const { InMemoryProfileStore } = require("../service/profileStore");
const { InMemoryEventLog } = require("../service/eventLog");
const { LearnedModel } = require("../model/learnedModel");
const { topTags } = require("../model/userProfile");

function printSlate(label, slate) {
  console.log(`\n── ${label} ───────────────────────────────────────────────`);
  for (const item of slate.items) {
    const score = item.finalScore.toFixed(2);
    const why = item.reasons.join(", ");
    console.log(`  [${item.source.padEnd(8)}] ${score}  ${item.title}`);
    console.log(`    ${item.channelName} · #${item.tags.join(" #")} · why: ${why}`);
  }
  console.log(`  pool=${slate.debug.candidatePoolSize} filtered=${slate.debug.afterFilter} slate=${slate.debug.slateSize}`);
}

function printProfile(label, profile) {
  const tags = topTags(profile, 5).map((t) => `${t.tag}:${t.v.toFixed(2)}`).join("  ");
  console.log(`\n${label} -> watches=${profile.watchHistory.length} likes=${profile.liked.length} skips=${profile.skipped.length}`);
  console.log(`  topTags: ${tags || "(none)"}`);
}

async function main() {
  const catalog = new Catalog(generateCatalog(240));
  const profileStore = new InMemoryProfileStore();
  const eventLog = new InMemoryEventLog();
  const learnedModel = LearnedModel.loadFromDisk();
  console.log(learnedModel
    ? `[demo] using learned model (auc=${learnedModel.meta?.metrics?.auc?.toFixed(3)} recall@10=${learnedModel.meta?.metrics?.recallAt10?.toFixed(3)})`
    : "[demo] no artifacts — using heuristic model. Run `node recsys/training/runTrain.js` first.");
  const rec = new Recommender({ catalog, profileStore, eventLog, learnedModel });

  // Brand-new online user — no MF embedding, ranker uses neutral mfScore.
  const userId = "u_demo";

  // 1. Cold start — no history, expect trending/fresh-heavy slate.
  let slate = rec.recommend(userId, { page: 1, pageSize: 6 });
  printSlate("Cold start (page 1)", slate);
  printProfile("Profile after cold start", profileStore.get(userId));

  // 2. User gets interested in AI / tech videos. Watch + like a few.
  console.log("\n>> Simulating session 1: user watches and likes AI/tech videos");
  const aiTechWatches = catalog.all()
    .filter((v) => v.tags.includes("ai") || v.tags.includes("tech"))
    .slice(0, 4);
  for (const v of aiTechWatches) {
    rec.ingestEvent({ userId, videoId: v.id, type: "watch", watchRatio: 0.9 });
  }
  rec.ingestEvent({ userId, videoId: aiTechWatches[0].id, type: "like" });

  slate = rec.recommend(userId, { page: 1, pageSize: 6 });
  printSlate("After AI/tech session (page 1)", slate);
  printProfile("Profile after AI/tech session", profileStore.get(userId));

  // 3. User explicitly says "not interested" in a fashion video.
  console.log("\n>> Simulating skip on fashion content");
  const fashionVideo = catalog.all().find((v) => v.tags.includes("fashion"));
  if (fashionVideo) {
    rec.ingestEvent({ userId, videoId: fashionVideo.id, type: "skip" });
  }

  // 4. Page through the next slate — should appear on a subsequent call,
  //    not retreat.
  slate = rec.recommend(userId, { page: 2, pageSize: 6 });
  printSlate("Page 2 with same profile", slate);

  // 5. Distribution check across sources for a deeper page.
  const big = rec.recommend(userId, { page: 1, pageSize: 30 });
  const counts = big.items.reduce((acc, x) => ((acc[x.source] = (acc[x.source] || 0) + 1), acc), {});
  console.log(`\nSource distribution in top-30: ${JSON.stringify(counts)}`);

  console.log(`\nEvents logged: ${eventLog.size()}`);

  // ── Known-user demo: pick a user from the trained pool so the full
  //    learned path (per-user MF score) is exercised.
  if (learnedModel) {
    const knownUserId = Object.keys(learnedModel.mf.userEmb)[0];
    if (knownUserId) {
      const knownSlate = rec.recommend(knownUserId, { page: 1, pageSize: 6 });
      printSlate(`Known trained user (${knownUserId})`, knownSlate);
      const counts = knownSlate.items.reduce(
        (acc, x) => ((acc[x.source] = (acc[x.source] || 0) + 1), acc), {},
      );
      console.log(`  source distribution: ${JSON.stringify(counts)}`);
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
