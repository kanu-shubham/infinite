// Sequential schema migration runner. Each migration declares its `from` and
// `to` version and an `up(adapter)` function. `runMigrations` picks a path
// from the adapter's current version to the target, applies migrations in
// order, and stamps the new version.

const REGISTRY = [];

export function registerMigration(migration) {
  if (
    !migration ||
    !Number.isFinite(migration.from) ||
    !Number.isFinite(migration.to) ||
    typeof migration.up !== "function"
  ) {
    throw new Error("registerMigration: invalid migration");
  }
  REGISTRY.push(migration);
  REGISTRY.sort((a, b) => a.from - b.from);
}

export function clearMigrations() {
  REGISTRY.length = 0;
}

export function listMigrations() {
  return REGISTRY.slice();
}

export async function runMigrations(adapter, targetVersion) {
  let current = await adapter.version();
  if (current === targetVersion) return { applied: [], from: current, to: current };
  if (current > targetVersion) {
    throw new Error(
      `runMigrations: stored version ${current} exceeds target ${targetVersion}`
    );
  }
  const applied = [];
  while (current < targetVersion) {
    const step = REGISTRY.find((m) => m.from === current);
    if (!step) {
      throw new Error(
        `runMigrations: no migration path from version ${current} (target ${targetVersion})`
      );
    }
    await step.up(adapter);
    current = step.to;
    applied.push(step);
    await adapter.setVersion(current);
  }
  return { applied, from: applied[0] ? applied[0].from : current, to: current };
}

// v0 -> v1: no-op. Establishes the migration scaffold so future upgrades have
// a starting point to register against.
registerMigration({
  from: 0,
  to: 1,
  async up(_adapter) {},
});
