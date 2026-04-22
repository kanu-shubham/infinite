import { StorageAdapter } from "./StorageAdapter";

// localStorage-backed adapter. Wraps the synchronous API behind async so the
// MemoryManager can treat all storage uniformly. Emits quota events via an
// optional `onQuotaExceeded` callback so the app can surface a warning instead
// of crashing.

const DEFAULT_NAMESPACE = "memory";
const VERSION_KEY = "__version";

export class LocalStorageAdapter extends StorageAdapter {
  constructor({ namespace = DEFAULT_NAMESPACE, onQuotaExceeded = null } = {}) {
    super();
    this.namespace = namespace;
    this.onQuotaExceeded = onQuotaExceeded;
    this._storage = resolveStorage();
  }

  _k(key) {
    return `${this.namespace}:${key}`;
  }

  async get(key) {
    if (!this._storage) return null;
    const raw = this._storage.getItem(this._k(key));
    if (raw == null) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  async set(key, value) {
    if (!this._storage) return;
    try {
      this._storage.setItem(this._k(key), JSON.stringify(value));
    } catch (err) {
      if (isQuotaError(err) && typeof this.onQuotaExceeded === "function") {
        this.onQuotaExceeded({ key, err });
        return;
      }
      throw err;
    }
  }

  async del(key) {
    if (!this._storage) return;
    this._storage.removeItem(this._k(key));
  }

  async list(prefix = "") {
    if (!this._storage) return [];
    const out = [];
    const fullPrefix = `${this.namespace}:${prefix}`;
    for (let i = 0; i < this._storage.length; i++) {
      const k = this._storage.key(i);
      if (k && k.startsWith(fullPrefix)) out.push(k.slice(this.namespace.length + 1));
    }
    return out;
  }

  async version() {
    const v = await this.get(VERSION_KEY);
    return typeof v === "number" ? v : 0;
  }

  async setVersion(n) {
    await this.set(VERSION_KEY, n);
  }
}

function resolveStorage() {
  try {
    if (typeof localStorage !== "undefined") return localStorage;
  } catch {
    // Accessing localStorage in sandboxed contexts can throw SecurityError.
  }
  return null;
}

function isQuotaError(err) {
  if (!err) return false;
  return (
    err.name === "QuotaExceededError" ||
    err.code === 22 ||
    err.code === 1014 /* Firefox NS_ERROR_DOM_QUOTA_REACHED */
  );
}

export default LocalStorageAdapter;
