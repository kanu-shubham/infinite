import { AdaptiveRetriever } from "../adaptive/AdaptiveRetriever";
import { createEpsilonGreedyBandit } from "../adaptive/bandit";
import { ShortTermMemory } from "../shortTermMemory";
import { LongTermMemory } from "../longTermMemory";
import { KnowledgeGraph } from "../knowledgeGraph/KnowledgeGraph";
import { MemoryAdapter } from "../storage/memoryAdapter";

async function makeRetriever(overrides = {}) {
  const stm = new ShortTermMemory();
  const ltm = new LongTermMemory({ storage: new MemoryAdapter() });
  const kg = new KnowledgeGraph();
  return new AdaptiveRetriever({
    stm,
    ltm,
    kg,
    warmupQueries: 0,
    ...overrides,
  });
}

describe("AdaptiveRetriever", () => {
  test("query returns ranked results with per-source signals", async () => {
    const ret = await makeRetriever();
    await ret.stm.add({ text: "search: paris spa", salience: 2 });
    await ret.ltm.remember({ text: "Paris boutique hotel with spa and pool" });
    ret.kg.addTriple("hotel:X", "located_in", "paris", {
      fromType: "hotel",
      toType: "location",
    });
    ret.kg.addTriple("hotel:X", "has_amenity", "spa", {
      fromType: "hotel",
      toType: "amenity",
    });
    const { results, weights, latencyMs } = await ret.query("paris spa", { k: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(weights).toEqual(
      expect.objectContaining({ stm: expect.any(Number), ltm: expect.any(Number) })
    );
    expect(typeof latencyMs).toBe("number");
  });

  test("feedback updates the bandit arm for the dominant signal", async () => {
    const bandit = createEpsilonGreedyBandit(["stm", "ltm", "kg", "recency"], {
      epsilon: 0,
      priorN: 1,
      priorMean: 0.5,
    });
    const ret = await makeRetriever({ bandit });
    await ret.ltm.remember({ text: "ocean view resort in Miami" });
    const { results } = await ret.query("ocean miami", { k: 3 });
    const ltmResult = results.find((r) =>
      r.signals.some((s) => s.source === "ltm" && s.score > 0)
    );
    expect(ltmResult).toBeTruthy();
    const before = bandit.snapshot().ltm.mean;
    ret.feedback({ resultId: ltmResult.id, reward: 1 });
    const after = bandit.snapshot().ltm.mean;
    expect(after).toBeGreaterThan(before);
  });

  test("warmup weights are used for the first N queries", async () => {
    const ret = await makeRetriever({
      warmupQueries: 3,
      warmupWeights: { stm: 0.25, ltm: 0.25, kg: 0.25, recency: 0.25 },
    });
    await ret.ltm.remember({ text: "anything" });
    const { weights } = await ret.query("anything", { k: 1 });
    expect(weights).toEqual({ stm: 0.25, ltm: 0.25, kg: 0.25, recency: 0.25 });
  });
});
