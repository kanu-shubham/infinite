// Event log.
//
// In production, client events are POSTed to the API and forwarded to a queue
// (Kafka/Pub-Sub). Online consumers update the profile store in near-real-
// time; offline consumers feed the training pipeline.
//
// This implementation is in-memory + an optional fan-out callback so a test
// harness can subscribe to events for assertions.

class InMemoryEventLog {
  constructor() {
    this.events = [];
    this.subscribers = new Set();
  }

  append(event) {
    const stamped = { ...event, ts: event.ts || Date.now() };
    this.events.push(stamped);
    for (const fn of this.subscribers) {
      try { fn(stamped); } catch (e) { /* never let a subscriber kill the log */ }
    }
    return stamped;
  }

  subscribe(fn) {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  // For offline jobs / tests.
  drain() {
    const out = this.events;
    this.events = [];
    return out;
  }

  size() { return this.events.length; }
}

module.exports = { InMemoryEventLog };
