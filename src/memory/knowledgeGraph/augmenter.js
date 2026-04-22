import { createHeuristicExtractor } from "./extractor";

// Glue layer: subscribes to `ltm:remember` events and feeds each episode
// through an extractor to augment the KG. Kept separate from both the KG
// and LTM so either can be used standalone.

export function createAugmenter({ graph, extractor = null } = {}) {
  if (!graph) throw new Error("createAugmenter: graph is required");
  const impl = extractor || createHeuristicExtractor();
  let unsub = null;

  function consume(episode) {
    const { nodes, edges } = impl.extract(episode);
    for (const n of nodes) graph.upsertNode(n);
    for (const e of edges) graph.upsertEdge(e);
    return { nodeCount: nodes.length, edgeCount: edges.length };
  }

  return {
    consume,
    attach(emitter) {
      if (unsub) unsub();
      unsub = emitter.on("ltm:remember", ({ episode }) => {
        try {
          consume(episode);
        } catch (err) {
          if (typeof console !== "undefined") {
            console.error("[memory] augmenter error", err);
          }
        }
      });
      return () => {
        if (unsub) {
          unsub();
          unsub = null;
        }
      };
    },
    detach() {
      if (unsub) {
        unsub();
        unsub = null;
      }
    },
  };
}

export default createAugmenter;
