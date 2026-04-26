// Profile store. Holds per-user profile state behind a small interface so the
// underlying backing can be swapped (in-memory for tests, Redis/Bigtable in
// production) without changing call sites.

const { emptyProfile } = require("../model/userProfile");

class InMemoryProfileStore {
  constructor() {
    this.profiles = new Map();
  }

  get(userId) {
    let p = this.profiles.get(userId);
    if (!p) {
      p = emptyProfile(userId);
      this.profiles.set(userId, p);
    }
    return p;
  }

  put(userId, profile) {
    this.profiles.set(userId, profile);
  }

  delete(userId) {
    this.profiles.delete(userId);
  }

  // Test helper.
  size() { return this.profiles.size; }
}

module.exports = { InMemoryProfileStore };
