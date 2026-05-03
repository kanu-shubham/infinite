// Training orchestrator.
//
// Pipeline:
//   1. Load catalogue.
//   2. Generate synthetic interactions (or in production: load logged events).
//   3. Train MF (BPR) on positives → user/item embeddings.
//   4. Train logistic-regression ranker on labeled events using MF score as
//      one of its features.
//   5. Evaluate (pointwise AUC + log-loss; top-K Recall + NDCG; random
//      baseline for sanity).
//   6. Persist artifacts to recsys/artifacts/ for the inference path to load.
//
// Run: node recsys/training/runTrain.js

const fs = require("fs");
const path = require("path");

const { generateCatalog, Catalog } = require("../data/catalog");
const { generate } = require("./dataGenerator");
const { trainBPR, serialiseMF } = require("./matrixFactorization");
const { trainLogReg } = require("./rankerTrainer");
const {
  evaluatePointwise,
  evaluateRetrieval,
  evaluateRandomBaseline,
} = require("./evaluate");

const ARTIFACT_DIR = path.join(__dirname, "..", "artifacts");

function main() {
  const t0 = Date.now();
  console.log("[1/5] Loading catalogue");
  const catalog = new Catalog(generateCatalog(240));
  console.log(`      ${catalog.size()} videos`);

  console.log("[2/5] Generating synthetic interactions");
  const { users, train, test } = generate(catalog, { numUsers: 200, perUser: 30 });
  const trainPos = train.filter((e) => e.label === 1).length;
  const testPos = test.filter((e) => e.label === 1).length;
  console.log(`      users=${users.length}  train=${train.length} (pos=${trainPos})  test=${test.length} (pos=${testPos})`);

  console.log("[3/5] Training BPR matrix factorisation");
  const mfModel = trainBPR(train, catalog, { dim: 12, epochs: 30, lr: 0.03, reg: 0.05 });

  console.log("[4/5] Training logistic-regression ranker");
  const rankerModel = trainLogReg(train, catalog, mfModel, { epochs: 25, lr: 0.1, reg: 0.001 });

  console.log("[5/5] Evaluating");
  const point = evaluatePointwise({ trainEvents: train, testEvents: test, catalog, mfModel, rankerModel });
  const retrieval10 = evaluateRetrieval({ trainEvents: train, testEvents: test, catalog, mfModel, rankerModel, k: 10 });
  const baseline10 = evaluateRandomBaseline({ trainEvents: train, testEvents: test, catalog, k: 10 });

  console.log("");
  console.log(`  Pointwise AUC          ${point.auc.toFixed(4)}    (n=${point.n})`);
  console.log(`  Pointwise log-loss     ${point.logloss.toFixed(4)}`);
  console.log(`  Recall@10              ${retrieval10.recallAtK.toFixed(4)}    (users=${retrieval10.users})`);
  console.log(`  NDCG@10                ${retrieval10.ndcgAtK.toFixed(4)}`);
  console.log(`  Random Recall@10       ${baseline10.recallAtK.toFixed(4)}    (baseline)`);

  // Persist.
  if (!fs.existsSync(ARTIFACT_DIR)) fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
  const mfPath = path.join(ARTIFACT_DIR, "mf.json");
  const rankerPath = path.join(ARTIFACT_DIR, "ranker.json");
  const metaPath = path.join(ARTIFACT_DIR, "meta.json");

  fs.writeFileSync(mfPath, JSON.stringify(serialiseMF(mfModel)));
  fs.writeFileSync(rankerPath, JSON.stringify(rankerModel));
  fs.writeFileSync(metaPath, JSON.stringify({
    trainedAt: new Date().toISOString(),
    catalogSize: catalog.size(),
    numUsers: users.length,
    metrics: {
      auc: point.auc,
      logloss: point.logloss,
      recallAt10: retrieval10.recallAtK,
      ndcgAt10: retrieval10.ndcgAtK,
      randomRecallAt10: baseline10.recallAtK,
    },
    elapsedMs: Date.now() - t0,
  }, null, 2));

  console.log("");
  console.log(`Artifacts written to ${ARTIFACT_DIR}`);
  console.log(`  mf.json      ${fs.statSync(mfPath).size} bytes`);
  console.log(`  ranker.json  ${fs.statSync(rankerPath).size} bytes`);
  console.log(`Total time: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

if (require.main === module) main();

module.exports = { main };
