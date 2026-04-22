import { StorageAdapter } from "./StorageAdapter";

// In-process storage adapter backed by a Map. The default when no persistence
// is requested; also used in tests.

export class MemoryAdapter extends StorageAdapter {
  constructor() {
    super();
    this._store = new Map();
    this._version = 0;
  }

  async get(key) {
    return this._store.has(key) ? this._store.get(key) : null;
  }

  async set(key, value) {
    this._store.set(key, value);
  }

  async del(key) {
    this._store.delete(key);
  }

  async list(prefix = "") {
    const out = [];
    for (const k of this._store.keys()) {
      if (!prefix || k.startsWith(prefix)) out.push(k);
    }
    return out;
  }

  async version() {
    return this._version;
  }

  async setVersion(n) {
    this._version = n;
  }
}

export default MemoryAdapter;
