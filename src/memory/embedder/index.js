import { createBagOfWordsEmbedder } from "./bagOfWordsEmbedder";
import { createTransformerEmbedder } from "./transformerEmbedder";

const REGISTRY = new Map();
let defaultFactory = createBagOfWordsEmbedder;

REGISTRY.set("bag-of-words", createBagOfWordsEmbedder);
REGISTRY.set("transformer", createTransformerEmbedder);

export function registerEmbedder(name, factory) {
  if (!name || typeof factory !== "function") {
    throw new Error("registerEmbedder: name and factory are required");
  }
  REGISTRY.set(name, factory);
}

export function getEmbedder(name, options) {
  const factory = REGISTRY.get(name);
  if (!factory) throw new Error(`Unknown embedder: ${name}`);
  return factory(options);
}

export function setDefaultEmbedder(factory) {
  if (typeof factory !== "function") {
    throw new Error("setDefaultEmbedder: factory must be a function");
  }
  defaultFactory = factory;
}

export function getDefaultEmbedder() {
  return defaultFactory();
}

export { createBagOfWordsEmbedder, createTransformerEmbedder };
