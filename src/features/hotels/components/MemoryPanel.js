import React, { useMemo, useState } from "react";
import {
  useShortTermMemory,
  useRecall,
  useKnowledgeGraph,
  useAdaptiveRetrieval,
  useMemoryMetrics,
} from "../../../memory";
import "./MemoryPanel.css";

const TABS = [
  { id: "recent", label: "Recent" },
  { id: "suggested", label: "Suggested" },
  { id: "graph", label: "Graph" },
  { id: "metrics", label: "Metrics" },
];

function timeAgo(ts) {
  const secs = Math.max(1, Math.round((Date.now() - ts) / 1000));
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  return `${Math.round(secs / 3600)}h ago`;
}

function RecentTab() {
  const { items } = useShortTermMemory();
  const reversed = items.slice().reverse();
  if (reversed.length === 0) {
    return <p className="memory-panel__empty">No context yet. Try a search.</p>;
  }
  return (
    <ul className="memory-panel__list">
      {reversed.map((item) => (
        <li key={item.id} className="memory-panel__item">
          <span className="memory-panel__item-text">{item.text}</span>
          <span className="memory-panel__item-meta">
            salience {item.salience.toFixed(1)} · {timeAgo(item.createdAt)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function SuggestedTab({ query }) {
  const { results, weights, latencyMs, isLoading } = useRecall(query, { k: 6 });
  if (isLoading) {
    return <p className="memory-panel__empty">Retrieving…</p>;
  }
  if (!query) {
    return (
      <p className="memory-panel__empty">
        Type or commit a filter to seed the retriever.
      </p>
    );
  }
  if (results.length === 0) {
    return <p className="memory-panel__empty">Nothing relevant yet.</p>;
  }
  return (
    <>
      <div className="memory-panel__kv">
        <span className="memory-panel__kv-label">Latency</span>
        <span className="memory-panel__kv-value">{latencyMs.toFixed(1)} ms</span>
      </div>
      <h4 className="memory-panel__section-title">Fusion weights</h4>
      <div className="memory-panel__weights">
        {weights &&
          Object.entries(weights).map(([arm, w]) => (
            <div key={arm} className="memory-panel__weight">
              <div className="memory-panel__weight-label">{arm}</div>
              <div className="memory-panel__weight-value">
                {(w * 100).toFixed(0)}%
              </div>
            </div>
          ))}
      </div>
      <h4 className="memory-panel__section-title">Top results</h4>
      <ul className="memory-panel__list">
        {results.map((r) => (
          <li key={r.id} className="memory-panel__item">
            <div>
              {r.signals.map((s) => (
                <span
                  key={`${r.id}-${s.source}`}
                  className={`memory-panel__source-tag memory-panel__source-tag--${s.source}`}
                >
                  {s.source}
                </span>
              ))}
            </div>
            <span className="memory-panel__item-text">
              {truncate(describePayload(r.payload), 80)}
            </span>
            <span className="memory-panel__item-meta">
              score {(r.score * 100).toFixed(1)}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}

function GraphTab({ query }) {
  const { size, graph, traverse } = useKnowledgeGraph();
  const hits = useMemo(() => {
    const tokens = String(query || "")
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    const seeds = [];
    for (const tok of tokens) {
      if (graph.getNode(`amenity:${tok}`)) seeds.push(`amenity:${tok}`);
      if (graph.getNode(`location:${tok}`)) seeds.push(`location:${tok}`);
    }
    if (!seeds.length) return [];
    return traverse(seeds, { maxDepth: 2, limit: 10, budgetMs: 20 });
  }, [graph, traverse, query]);

  return (
    <>
      <div className="memory-panel__kv">
        <span className="memory-panel__kv-label">Nodes</span>
        <span className="memory-panel__kv-value">{size.nodes}</span>
        <span className="memory-panel__kv-label">Edges</span>
        <span className="memory-panel__kv-value">{size.edges}</span>
      </div>
      {hits.length === 0 ? (
        <p className="memory-panel__empty">
          Save hotels or type an amenity / location in search to populate the graph.
        </p>
      ) : (
        <>
          <h4 className="memory-panel__section-title">Neighbors of "{query}"</h4>
          <ul className="memory-panel__list">
            {hits.map((h) => (
              <li key={h.node.id} className="memory-panel__item">
                <span className="memory-panel__item-text">
                  {h.node.label} <small>({h.node.type})</small>
                </span>
                <span className="memory-panel__item-meta">
                  depth {h.depth} · weight {h.weight.toFixed(1)}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}

function MetricsTab() {
  const snap = useMemoryMetrics();
  const { weights, queries, resetBandit } = useAdaptiveRetrieval();
  return (
    <>
      <div className="memory-panel__kv">
        <span className="memory-panel__kv-label">STM size</span>
        <span className="memory-panel__kv-value">{snap.stmSize}</span>
        <span className="memory-panel__kv-label">LTM size</span>
        <span className="memory-panel__kv-value">{snap.ltmSize}</span>
        <span className="memory-panel__kv-label">KG nodes</span>
        <span className="memory-panel__kv-value">{snap.kgNodes}</span>
        <span className="memory-panel__kv-label">KG edges</span>
        <span className="memory-panel__kv-value">{snap.kgEdges}</span>
        <span className="memory-panel__kv-label">Queries</span>
        <span className="memory-panel__kv-value">{queries}</span>
        <span className="memory-panel__kv-label">Recall p50</span>
        <span className="memory-panel__kv-value">
          {snap.latency ? snap.latency.p50.toFixed(1) : 0} ms
        </span>
        <span className="memory-panel__kv-label">Recall p95</span>
        <span className="memory-panel__kv-value">
          {snap.latency ? snap.latency.p95.toFixed(1) : 0} ms
        </span>
      </div>
      <h4 className="memory-panel__section-title">Bandit weights</h4>
      <div className="memory-panel__weights">
        {weights &&
          Object.entries(weights).map(([arm, w]) => (
            <div key={arm} className="memory-panel__weight">
              <div className="memory-panel__weight-label">{arm}</div>
              <div className="memory-panel__weight-value">
                {(w * 100).toFixed(0)}%
              </div>
            </div>
          ))}
      </div>
      <div className="memory-panel__footer" style={{ padding: "12px 0 0" }}>
        <span>Schema v1</span>
        <button type="button" onClick={resetBandit}>
          Reset bandit
        </button>
      </div>
    </>
  );
}

export default function MemoryPanel({ query }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("recent");

  return (
    <>
      <button
        type="button"
        className="memory-panel__toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? "Close memory" : "Memory"}
      </button>
      <aside
        className={`memory-panel${open ? " memory-panel--open" : ""}`}
        aria-hidden={!open}
      >
        <header className="memory-panel__header">
          <h3 className="memory-panel__title">Memory</h3>
          <button
            type="button"
            className="memory-panel__close"
            onClick={() => setOpen(false)}
            aria-label="Close"
          >
            ×
          </button>
        </header>
        <nav className="memory-panel__tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`memory-panel__tab${
                tab === t.id ? " memory-panel__tab--active" : ""
              }`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="memory-panel__body">
          {tab === "recent" && <RecentTab />}
          {tab === "suggested" && <SuggestedTab query={query} />}
          {tab === "graph" && <GraphTab query={query} />}
          {tab === "metrics" && <MetricsTab />}
        </div>
      </aside>
    </>
  );
}

function describePayload(payload) {
  if (!payload) return "(no payload)";
  if (typeof payload === "string") return payload;
  if (payload.text) return payload.text;
  if (payload.label) return payload.label;
  return JSON.stringify(payload);
}

function truncate(s, n) {
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1)}…`;
}
