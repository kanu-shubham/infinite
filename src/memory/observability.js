// Aggregates events from an EventEmitter into a lightweight metrics object
// (counts, rolling latency percentiles, current sizes). Exposed via a
// subscribe() surface so UI and external telemetry can tap in.

const DEFAULT_WINDOW = 100;

export function createMetricsCollector(emitter, { window = DEFAULT_WINDOW } = {}) {
  const counts = Object.create(null);
  const latencies = [];
  let maxLatency = 0;
  let stmSize = 0;
  let ltmSize = 0;
  let kgNodes = 0;
  let kgEdges = 0;
  const subs = new Set();

  function bump(event) {
    counts[event] = (counts[event] || 0) + 1;
  }

  function snapshot() {
    return {
      counts: { ...counts },
      latency: summarize(latencies),
      maxLatency,
      stmSize,
      ltmSize,
      kgNodes,
      kgEdges,
    };
  }

  function emitChange() {
    const snap = snapshot();
    for (const fn of subs) {
      try {
        fn(snap);
      } catch (err) {
        if (typeof console !== "undefined") console.error("[metrics] listener", err);
      }
    }
  }

  const offs = [];
  offs.push(emitter.on("stm:add", () => { bump("stm:add"); stmSize += 1; emitChange(); }));
  offs.push(
    emitter.on("stm:evict", ({ items = [] }) => {
      bump("stm:evict");
      stmSize = Math.max(0, stmSize - items.length);
      emitChange();
    })
  );
  offs.push(emitter.on("stm:clear", () => { bump("stm:clear"); stmSize = 0; emitChange(); }));
  offs.push(emitter.on("ltm:remember", () => { bump("ltm:remember"); ltmSize += 1; emitChange(); }));
  offs.push(emitter.on("ltm:forget", () => { bump("ltm:forget"); ltmSize = Math.max(0, ltmSize - 1); emitChange(); }));
  offs.push(
    emitter.on("ltm:recall", () => {
      bump("ltm:recall");
      emitChange();
    })
  );
  offs.push(
    emitter.on("kg:upsert", ({ kind }) => {
      bump(`kg:upsert:${kind}`);
      if (kind === "node") kgNodes += 1;
      if (kind === "edge") kgEdges += 1;
      emitChange();
    })
  );
  offs.push(
    emitter.on("retrieval:done", ({ latencyMs }) => {
      bump("retrieval:done");
      if (Number.isFinite(latencyMs)) {
        latencies.push(latencyMs);
        if (latencies.length > window) latencies.shift();
        if (latencyMs > maxLatency) maxLatency = latencyMs;
      }
      emitChange();
    })
  );
  offs.push(emitter.on("feedback:applied", () => { bump("feedback:applied"); emitChange(); }));
  offs.push(emitter.on("decay:run", () => { bump("decay:run"); emitChange(); }));
  offs.push(emitter.on("storage:quotaExceeded", () => { bump("storage:quotaExceeded"); emitChange(); }));

  return {
    snapshot,
    subscribe(fn) {
      subs.add(fn);
      fn(snapshot());
      return () => subs.delete(fn);
    },
    seed({ stm = 0, ltm = 0, kgN = 0, kgE = 0 } = {}) {
      stmSize = stm;
      ltmSize = ltm;
      kgNodes = kgN;
      kgEdges = kgE;
      emitChange();
    },
    dispose() {
      for (const off of offs) off();
      subs.clear();
    },
  };
}

function summarize(samples) {
  if (!samples.length) return { count: 0, p50: 0, p95: 0, mean: 0 };
  const sorted = samples.slice().sort((a, b) => a - b);
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const pick = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  return {
    count: samples.length,
    mean,
    p50: pick(0.5),
    p95: pick(0.95),
  };
}
