import { ShortTermMemory } from "../shortTermMemory";
import { EventEmitter } from "../utils";

describe("ShortTermMemory", () => {
  test("respects capacity and evicts the oldest lowest-salience entry", async () => {
    const emitter = new EventEmitter();
    const evicted = [];
    emitter.on("stm:evict", ({ items }) => evicted.push(...items));
    const stm = new ShortTermMemory({ capacity: 3, emitter });

    await stm.add({ text: "one", salience: 1 });
    await stm.add({ text: "two", salience: 1 });
    await stm.add({ text: "three", salience: 1 });
    await stm.add({ text: "four", salience: 1 });

    const texts = stm.window().map((i) => i.text);
    expect(texts).toEqual(["two", "three", "four"]);
    expect(evicted.map((i) => i.text)).toEqual(["one"]);
  });

  test("drops entries below salience floor immediately", async () => {
    const stm = new ShortTermMemory({ capacity: 10, salienceFloor: 0.5 });
    await stm.add({ text: "kept", salience: 1 });
    await stm.add({ text: "dropped", salience: 0.1 });
    expect(stm.window().map((i) => i.text)).toEqual(["kept"]);
  });

  test("rank orders by recency × salience", async () => {
    const stm = new ShortTermMemory({ capacity: 5 });
    await stm.add({ text: "low-old", salience: 1 });
    await stm.add({ text: "high-new", salience: 5 });
    await stm.add({ text: "low-new", salience: 1 });
    const ranked = stm.rank().map((i) => i.text);
    expect(ranked[0]).toBe("high-new");
  });

  test("boost raises salience of an entry", async () => {
    const stm = new ShortTermMemory();
    const item = await stm.add({ text: "x", salience: 1 });
    stm.boost(item.id, 2);
    expect(stm.window()[0].salience).toBe(3);
  });
});
