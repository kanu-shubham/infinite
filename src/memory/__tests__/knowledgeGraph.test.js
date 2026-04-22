import { KnowledgeGraph } from "../knowledgeGraph/KnowledgeGraph";
import { createAugmenter } from "../knowledgeGraph/augmenter";
import { EventEmitter } from "../utils";

describe("KnowledgeGraph", () => {
  test("upsert merges props and accumulates weight", () => {
    const kg = new KnowledgeGraph();
    kg.upsertNode({ id: "hotel:1", type: "hotel", label: "Palace", props: { stars: 5 } });
    kg.upsertNode({ id: "hotel:1", props: { city: "Rome" } });
    const node = kg.getNode("hotel:1");
    expect(node.props).toEqual({ stars: 5, city: "Rome" });
    expect(node.weight).toBe(2);
  });

  test("traverse performs weighted BFS respecting depth", () => {
    const kg = new KnowledgeGraph();
    kg.upsertNode({ id: "hotel:1", type: "hotel", label: "Palace" });
    kg.upsertNode({ id: "amenity:pool", type: "amenity", label: "pool" });
    kg.upsertNode({ id: "amenity:spa", type: "amenity", label: "spa" });
    kg.upsertNode({ id: "location:rome", type: "location", label: "rome" });
    kg.upsertEdge({ from: "hotel:1", to: "amenity:pool", relation: "has_amenity" });
    kg.upsertEdge({ from: "hotel:1", to: "location:rome", relation: "located_in" });
    kg.upsertEdge({ from: "amenity:pool", to: "amenity:spa", relation: "similar_to" });

    const visited = kg.traverse(["hotel:1"], { maxDepth: 2, limit: 10, budgetMs: 100 });
    const ids = visited.map((v) => v.node.id);
    expect(ids).toEqual(expect.arrayContaining(["amenity:pool", "location:rome"]));
    const spa = visited.find((v) => v.node.id === "amenity:spa");
    expect(spa && spa.depth).toBe(2);
  });

  test("augmenter consumes an LTM remember event and upserts triples", () => {
    const emitter = new EventEmitter();
    const kg = new KnowledgeGraph({ emitter });
    const augmenter = createAugmenter({ graph: kg });
    augmenter.attach(emitter);

    emitter.emit("ltm:remember", {
      episode: {
        id: "ep1",
        text: "Pool and spa in Rome",
        meta: {
          subjectId: "hotel:1",
          subjectType: "hotel",
          subjectLabel: "Palace",
          location: "Rome",
          amenities: ["pool", "spa"],
        },
      },
    });

    const neighbors = kg.neighbors("hotel:1");
    const labels = neighbors.map((n) => n.node.id).sort();
    expect(labels).toEqual(
      expect.arrayContaining(["amenity:pool", "amenity:spa", "location:rome"])
    );
  });
});
