import { createMemoryManager } from "../MemoryManager";
import { MemoryAdapter } from "../storage/memoryAdapter";

describe("MemoryManager integration", () => {
  test("observe writes into STM and persists salient events into LTM", async () => {
    const mgr = createMemoryManager({ storage: new MemoryAdapter() });
    await mgr.init();
    await mgr.observe({
      text: "search for paris spa",
      salience: 2,
      persist: true,
    });
    expect(mgr.stm.size()).toBe(1);
    expect(mgr.ltm.size()).toBe(1);
    await mgr.dispose();
  });

  test("query routes through the retriever and emits retrieval:done", async () => {
    const mgr = createMemoryManager({ storage: new MemoryAdapter() });
    await mgr.init();
    const events = [];
    mgr.on("retrieval:done", (e) => events.push(e));
    await mgr.observe({
      text: "Paris hotel with spa and pool",
      salience: 2,
      persist: true,
      subjectId: "hotel:1",
      subjectType: "hotel",
      subjectLabel: "Test Palace",
      meta: { location: "paris", amenities: ["spa", "pool"] },
    });
    const { results, latencyMs } = await mgr.query("paris spa");
    expect(results.length).toBeGreaterThan(0);
    expect(events.length).toBe(1);
    expect(typeof latencyMs).toBe("number");
    await mgr.dispose();
  });

  test("augmenter wires KG updates off LTM remember events", async () => {
    const mgr = createMemoryManager({ storage: new MemoryAdapter() });
    await mgr.init();
    await mgr.ltm.remember({
      text: "spa and sauna in rome",
      meta: {
        subjectId: "hotel:42",
        subjectType: "hotel",
        subjectLabel: "Roma Grand",
        location: "rome",
        amenities: ["spa", "sauna"],
      },
    });
    // Let the emitter tick.
    await new Promise((r) => setTimeout(r, 0));
    const { nodes, edges } = mgr.kg.size();
    expect(nodes).toBeGreaterThan(0);
    expect(edges).toBeGreaterThan(0);
    await mgr.dispose();
  });

  test("metrics snapshot reflects subsystem state", async () => {
    const mgr = createMemoryManager({ storage: new MemoryAdapter() });
    await mgr.init();
    await mgr.observe({ text: "hi", salience: 2, persist: true });
    const snap = mgr.metrics.snapshot();
    expect(snap.stmSize).toBe(1);
    expect(snap.ltmSize).toBe(1);
    await mgr.dispose();
  });
});
