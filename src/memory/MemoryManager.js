import { EventEmitter, SingleFlightQueue } from "./utils";
import { createBagOfWordsEmbedder } from "./embedder/bagOfWordsEmbedder";
import { MemoryAdapter } from "./storage/memoryAdapter";
import { runMigrations } from "./storage/migrations";
import { ShortTermMemory } from "./shortTermMemory";
import { LongTermMemory } from "./longTermMemory";
import { KnowledgeGraph } from "./knowledgeGraph/KnowledgeGraph";
import { createAugmenter } from "./knowledgeGraph/augmenter";
import { AdaptiveRetriever } from "./adaptive/AdaptiveRetriever";
import { createMetricsCollector } from "./observability";

const SCHEMA_VERSION = 1;

// Orchestrates the four memory subsystems. Consumers talk to this surface;
// the underlying modules can still be used standalone. Lifecycle:
//   const mem = createMemoryManager({ storage, embedder });
//   await mem.init();
//   ...
//   await mem.dispose();

export function createMemoryManager(config = {}) {
  const emitter = config.emitter || new EventEmitter();
  const queue = config.queue || new SingleFlightQueue();
  const storage = config.storage || new MemoryAdapter();
  const embedder = config.embedder || createBagOfWordsEmbedder();

  const stm = new ShortTermMemory({ embedder, emitter, ...(config.stm || {}) });
  const ltm = new LongTermMemory({
    embedder,
    storage,
    emitter,
    queue,
    ...(config.ltm || {}),
  });
  const kg = new KnowledgeGraph({
    storage,
    emitter,
    queue,
    ...(config.kg || {}),
  });
  const augmenter = createAugmenter({
    graph: kg,
    extractor: config.extractor || null,
  });
  const retriever = new AdaptiveRetriever({
    stm,
    ltm,
    kg,
    emitter,
    ...(config.retriever || {}),
  });
  const metrics = createMetricsCollector(emitter, {
    window: config.metricsWindow,
  });

  let detachAugmenter = null;
  let ready = false;
  let initPromise = null;

  async function init() {
    if (ready) return;
    if (initPromise) return initPromise;
    initPromise = (async () => {
      await storage.open();
      await runMigrations(storage, SCHEMA_VERSION);
      await ltm.init();
      await kg.init();
      detachAugmenter = augmenter.attach(emitter);
      metrics.seed({
        stm: stm.size(),
        ltm: ltm.size(),
        kgN: kg.size().nodes,
        kgE: kg.size().edges,
      });
      ready = true;
      emitter.emit("manager:ready", {});
    })();
    return initPromise;
  }

  async function observe(event) {
    if (!ready) await init();
    const salience = event.salience != null ? event.salience : 1;
    const item = await stm.add({
      text: event.text,
      role: event.role,
      salience,
      meta: event.meta,
    });
    const persist =
      event.persist === true ||
      salience >= (config.persistThreshold != null ? config.persistThreshold : 1.5);
    let episode = null;
    if (persist) {
      episode = await ltm.remember({
        id: event.id,
        text: event.text,
        tags: event.tags || [],
        meta: { ...event.meta, subjectId: event.subjectId, subjectLabel: event.subjectLabel, subjectType: event.subjectType },
      });
    }
    return { item, episode };
  }

  async function query(text, options) {
    if (!ready) await init();
    return retriever.query(text, options);
  }

  function feedback(payload) {
    return retriever.feedback(payload);
  }

  async function consolidate() {
    if (!ready) await init();
    return ltm.consolidate();
  }

  async function dispose() {
    if (detachAugmenter) {
      detachAugmenter();
      detachAugmenter = null;
    }
    metrics.dispose();
    await storage.close();
    emitter.emit("manager:disposed", {});
    emitter.removeAll();
    ready = false;
    initPromise = null;
  }

  return {
    // Lifecycle
    init,
    dispose,
    get ready() {
      return ready;
    },
    // Subsystems (escape hatch)
    stm,
    ltm,
    kg,
    retriever,
    augmenter,
    // High-level API
    observe,
    query,
    feedback,
    consolidate,
    // Observability
    on: (ev, fn) => emitter.on(ev, fn),
    off: (ev, fn) => emitter.off(ev, fn),
    metrics,
    get schemaVersion() {
      return SCHEMA_VERSION;
    },
  };
}

export default createMemoryManager;
