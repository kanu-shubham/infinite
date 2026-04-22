// Abstract storage contract. Every concrete adapter must provide an async,
// version-aware surface so the MemoryManager can run migrations without
// knowing where data actually lives.

/* eslint-disable no-unused-vars */
/** @implements {import("../types").StorageAdapter} */
export class StorageAdapter {
  async open() {}
  async get(key) {
    throw new Error("StorageAdapter.get: not implemented");
  }
  async set(key, value) {
    throw new Error("StorageAdapter.set: not implemented");
  }
  async del(key) {
    throw new Error("StorageAdapter.del: not implemented");
  }
  async list(prefix = "") {
    throw new Error("StorageAdapter.list: not implemented");
  }
  async version() {
    return 0;
  }
  async setVersion(n) {
    throw new Error("StorageAdapter.setVersion: not implemented");
  }
  async close() {}
}

export class NoopAdapter extends StorageAdapter {
  async get() {
    return null;
  }
  async set() {}
  async del() {}
  async list() {
    return [];
  }
  async setVersion() {}
}
