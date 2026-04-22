export { createMemoryManager } from "./MemoryManager";
export { ShortTermMemory } from "./shortTermMemory";
export { LongTermMemory } from "./longTermMemory";
export { KnowledgeGraph } from "./knowledgeGraph/KnowledgeGraph";
export { createHeuristicExtractor } from "./knowledgeGraph/extractor";
export { createAugmenter } from "./knowledgeGraph/augmenter";
export { AdaptiveRetriever } from "./adaptive/AdaptiveRetriever";
export { createEpsilonGreedyBandit } from "./adaptive/bandit";
export {
  normalizeSTM,
  normalizeLTM,
  normalizeKG,
  normalizeRecency,
} from "./adaptive/signals";
export {
  createBagOfWordsEmbedder,
  createTransformerEmbedder,
  registerEmbedder,
  getEmbedder,
  setDefaultEmbedder,
  getDefaultEmbedder,
} from "./embedder";
export { StorageAdapter, NoopAdapter } from "./storage/StorageAdapter";
export { MemoryAdapter } from "./storage/memoryAdapter";
export { LocalStorageAdapter } from "./storage/localStorageAdapter";
export {
  runMigrations,
  registerMigration,
  listMigrations,
  clearMigrations,
} from "./storage/migrations";
export { createMetricsCollector } from "./observability";
export {
  EventEmitter,
  SingleFlightQueue,
  featureVector,
  tokenize,
  cosineSimilarity,
  decayStrength,
  now,
  uid,
} from "./utils";

export { MemoryProvider, useMemory } from "./react/MemoryProvider";
export { useShortTermMemory } from "./react/useShortTermMemory";
export { useRecall } from "./react/useRecall";
export { useKnowledgeGraph } from "./react/useKnowledgeGraph";
export { useAdaptiveRetrieval } from "./react/useAdaptiveRetrieval";
export { useMemoryMetrics } from "./react/useMemoryMetrics";
