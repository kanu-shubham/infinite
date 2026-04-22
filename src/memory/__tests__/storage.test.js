import { MemoryAdapter } from "../storage/memoryAdapter";
import {
  clearMigrations,
  listMigrations,
  registerMigration,
  runMigrations,
} from "../storage/migrations";

describe("storage", () => {
  test("MemoryAdapter round-trips values and lists prefixes", async () => {
    const s = new MemoryAdapter();
    await s.set("ltm.episodes", [1, 2, 3]);
    await s.set("kg.graph", { nodes: [] });
    expect(await s.get("ltm.episodes")).toEqual([1, 2, 3]);
    expect(await s.list("ltm.")).toEqual(["ltm.episodes"]);
    await s.del("ltm.episodes");
    expect(await s.get("ltm.episodes")).toBeNull();
  });

  test("runMigrations walks from current version up to target", async () => {
    const saved = listMigrations();
    clearMigrations();
    registerMigration({
      from: 0,
      to: 1,
      async up(adapter) {
        await adapter.set("marker.v1", true);
      },
    });
    registerMigration({
      from: 1,
      to: 2,
      async up(adapter) {
        await adapter.set("marker.v2", true);
      },
    });

    const adapter = new MemoryAdapter();
    const result = await runMigrations(adapter, 2);
    expect(result.applied.map((m) => m.to)).toEqual([1, 2]);
    expect(await adapter.version()).toBe(2);
    expect(await adapter.get("marker.v1")).toBe(true);
    expect(await adapter.get("marker.v2")).toBe(true);

    // Restore the default registry to avoid leaking into other suites.
    clearMigrations();
    for (const m of saved) registerMigration(m);
  });

  test("runMigrations throws when the stored version exceeds the target", async () => {
    const adapter = new MemoryAdapter();
    await adapter.setVersion(5);
    await expect(runMigrations(adapter, 2)).rejects.toThrow(/exceeds/);
  });
});
