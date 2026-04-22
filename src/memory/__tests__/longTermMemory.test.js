import { LongTermMemory } from "../longTermMemory";
import { MemoryAdapter } from "../storage/memoryAdapter";

describe("LongTermMemory", () => {
  test("remember then recall returns the episode and reinforces it", async () => {
    const ltm = new LongTermMemory({ storage: new MemoryAdapter() });
    await ltm.remember({ text: "luxury ocean view suite in Miami" });
    const hits = await ltm.recall("ocean miami");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].episode.accessCount).toBe(1);
    expect(hits[0].similarity).toBeGreaterThan(0);
  });

  test("persists to and reloads from the storage adapter", async () => {
    const storage = new MemoryAdapter();
    const a = new LongTermMemory({ storage });
    await a.remember({ text: "boutique hotel paris spa" });
    // Give the SingleFlightQueue a chance to flush.
    await new Promise((r) => setTimeout(r, 0));

    const b = new LongTermMemory({ storage });
    await b.init();
    const hits = await b.recall("paris spa");
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].episode.text).toMatch(/paris/);
  });

  test("consolidate drops sub-threshold episodes", async () => {
    const ltm = new LongTermMemory({
      storage: new MemoryAdapter(),
      halfLifeMs: 1, // effectively instant decay
      minStrength: 0.5,
    });
    await ltm.remember({ text: "ephemeral observation", strength: 0.6 });
    await new Promise((r) => setTimeout(r, 5));
    const dropped = await ltm.consolidate();
    expect(dropped).toBeGreaterThan(0);
    expect(ltm.size()).toBe(0);
  });
});
