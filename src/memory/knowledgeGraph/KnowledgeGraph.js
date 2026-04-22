import { EventEmitter, SingleFlightQueue, now, uid } from "../utils";
import { MemoryAdapter } from "../storage/memoryAdapter";

// A lightweight in-memory knowledge graph with optional async persistence.
// Supports weighted nodes and edges, neighborhood queries, weighted BFS
// traversal with a soft time budget, and JSON serialization.

const DEFAULTS = {
  maxNodes: 10000,
  maxEdges: 50000,
  storageKey: "kg.graph",
  edgeDecayRate: 0,
};

export class KnowledgeGraph {
  constructor(options = {}) {
    this.opts = { ...DEFAULTS, ...options };
    this.storage = options.storage || new MemoryAdapter();
    this.emitter = options.emitter || new EventEmitter();
    this.queue = options.queue || new SingleFlightQueue();
    this.nodes = new Map();
    this.edges = new Map();
    this._adj = new Map();
    this._reverseAdj = new Map();
    this._loaded = false;
  }

  async init() {
    if (this._loaded) return;
    await this.storage.open();
    const raw = await this.storage.get(this.opts.storageKey);
    if (raw && Array.isArray(raw.nodes) && Array.isArray(raw.edges)) {
      for (const n of raw.nodes) this._installNode(n);
      for (const e of raw.edges) this._installEdge(e);
    }
    this._loaded = true;
  }

  upsertNode(node) {
    const id = node.id || `${node.type || "node"}:${normalize(node.label)}`;
    const existing = this.nodes.get(id);
    const t = now();
    const merged = {
      id,
      type: node.type || (existing && existing.type) || "entity",
      label: node.label || (existing && existing.label) || id,
      props: { ...(existing ? existing.props : {}), ...(node.props || {}) },
      weight: existing
        ? existing.weight + (node.weight != null ? node.weight : 1)
        : node.weight != null
          ? node.weight
          : 1,
      createdAt: existing ? existing.createdAt : t,
      updatedAt: t,
    };
    this.nodes.set(id, merged);
    if (!this._adj.has(id)) this._adj.set(id, new Set());
    if (!this._reverseAdj.has(id)) this._reverseAdj.set(id, new Set());
    this._enforceNodeCap();
    this.emitter.emit("kg:upsert", { kind: "node", node: merged });
    this._persist();
    return merged;
  }

  upsertEdge(edge) {
    if (!edge || !edge.from || !edge.to || !edge.relation) {
      throw new Error("upsertEdge: from, to, relation required");
    }
    if (!this.nodes.has(edge.from)) this.upsertNode({ id: edge.from, label: edge.from });
    if (!this.nodes.has(edge.to)) this.upsertNode({ id: edge.to, label: edge.to });
    const key = edgeKey(edge.from, edge.to, edge.relation);
    const existing = this.edges.get(key);
    const t = now();
    const merged = {
      id: existing ? existing.id : edge.id || uid("edge"),
      from: edge.from,
      to: edge.to,
      relation: edge.relation,
      props: { ...(existing ? existing.props : {}), ...(edge.props || {}) },
      weight: existing
        ? existing.weight + (edge.weight != null ? edge.weight : 1)
        : edge.weight != null
          ? edge.weight
          : 1,
      createdAt: existing ? existing.createdAt : t,
      updatedAt: t,
    };
    this.edges.set(key, merged);
    this._adj.get(edge.from).add(key);
    this._reverseAdj.get(edge.to).add(key);
    this._enforceEdgeCap();
    this.emitter.emit("kg:upsert", { kind: "edge", edge: merged });
    this._persist();
    return merged;
  }

  addTriple(fromLabel, relation, toLabel, { fromType, toType, weight = 1 } = {}) {
    const from = this.upsertNode({ type: fromType, label: fromLabel });
    const to = this.upsertNode({ type: toType, label: toLabel });
    return this.upsertEdge({ from: from.id, to: to.id, relation, weight });
  }

  getNode(id) {
    return this.nodes.get(id) || null;
  }

  neighbors(id, { relation = null, direction = "out", minWeight = 0 } = {}) {
    const out = [];
    const seen = new Set();
    const sets = [];
    if (direction === "out" || direction === "both") sets.push(this._adj.get(id));
    if (direction === "in" || direction === "both") sets.push(this._reverseAdj.get(id));
    for (const set of sets) {
      if (!set) continue;
      for (const key of set) {
        const edge = this.edges.get(key);
        if (!edge) continue;
        if (relation && edge.relation !== relation) continue;
        if (edge.weight < minWeight) continue;
        const otherId = edge.from === id ? edge.to : edge.from;
        if (seen.has(otherId)) continue;
        seen.add(otherId);
        const node = this.nodes.get(otherId);
        if (node) out.push({ node, edge });
      }
    }
    out.sort((a, b) => b.edge.weight - a.edge.weight);
    return out;
  }

  // Weighted BFS from seed nodes. Visits nodes in order of cumulative edge
  // weight, bounded by depth and a soft wall-clock budget (ms).
  traverse(seedIds, { maxDepth = 2, minWeight = 0, budgetMs = 50, limit = 50 } = {}) {
    const start = now();
    const seeds = Array.isArray(seedIds) ? seedIds : [seedIds];
    const visited = new Map();
    const frontier = [];
    for (const id of seeds) {
      if (!this.nodes.has(id)) continue;
      visited.set(id, { depth: 0, weight: 0 });
      frontier.push({ id, depth: 0, weight: 0 });
    }
    const out = [];
    while (frontier.length) {
      if (now() - start > budgetMs) break;
      if (out.length >= limit) break;
      frontier.sort((a, b) => b.weight - a.weight);
      const { id, depth, weight } = frontier.shift();
      const node = this.nodes.get(id);
      if (!node) continue;
      out.push({ node, depth, weight });
      if (depth >= maxDepth) continue;
      for (const { node: nbr, edge } of this.neighbors(id, {
        direction: "both",
        minWeight,
      })) {
        const cum = weight + edge.weight;
        const prev = visited.get(nbr.id);
        if (prev && prev.weight >= cum) continue;
        visited.set(nbr.id, { depth: depth + 1, weight: cum });
        frontier.push({ id: nbr.id, depth: depth + 1, weight: cum });
      }
    }
    return out.slice(1); // drop seed itself
  }

  prune(minNodeWeight = 0.5, minEdgeWeight = 0.5) {
    let removedEdges = 0;
    for (const [key, edge] of this.edges) {
      if (edge.weight < minEdgeWeight) {
        this.edges.delete(key);
        const a = this._adj.get(edge.from);
        const b = this._reverseAdj.get(edge.to);
        if (a) a.delete(key);
        if (b) b.delete(key);
        removedEdges++;
      }
    }
    let removedNodes = 0;
    for (const [id, node] of this.nodes) {
      const outDeg = (this._adj.get(id) || new Set()).size;
      const inDeg = (this._reverseAdj.get(id) || new Set()).size;
      if (node.weight < minNodeWeight && outDeg === 0 && inDeg === 0) {
        this.nodes.delete(id);
        this._adj.delete(id);
        this._reverseAdj.delete(id);
        removedNodes++;
      }
    }
    if (removedNodes || removedEdges) this._persist();
    return { removedNodes, removedEdges };
  }

  size() {
    return { nodes: this.nodes.size, edges: this.edges.size };
  }

  toJSON() {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: Array.from(this.edges.values()),
    };
  }

  _installNode(n) {
    this.nodes.set(n.id, n);
    if (!this._adj.has(n.id)) this._adj.set(n.id, new Set());
    if (!this._reverseAdj.has(n.id)) this._reverseAdj.set(n.id, new Set());
  }

  _installEdge(e) {
    const key = edgeKey(e.from, e.to, e.relation);
    this.edges.set(key, e);
    if (!this._adj.has(e.from)) this._adj.set(e.from, new Set());
    if (!this._reverseAdj.has(e.to)) this._reverseAdj.set(e.to, new Set());
    this._adj.get(e.from).add(key);
    this._reverseAdj.get(e.to).add(key);
  }

  _enforceNodeCap() {
    if (this.nodes.size <= this.opts.maxNodes) return;
    const entries = Array.from(this.nodes.values()).sort(
      (a, b) => a.weight - b.weight
    );
    while (this.nodes.size > this.opts.maxNodes && entries.length) {
      const victim = entries.shift();
      this.nodes.delete(victim.id);
      const outSet = this._adj.get(victim.id);
      if (outSet) for (const k of outSet) this.edges.delete(k);
      const inSet = this._reverseAdj.get(victim.id);
      if (inSet) for (const k of inSet) this.edges.delete(k);
      this._adj.delete(victim.id);
      this._reverseAdj.delete(victim.id);
    }
  }

  _enforceEdgeCap() {
    if (this.edges.size <= this.opts.maxEdges) return;
    const entries = Array.from(this.edges.entries()).sort(
      (a, b) => a[1].weight - b[1].weight
    );
    while (this.edges.size > this.opts.maxEdges && entries.length) {
      const [key, edge] = entries.shift();
      this.edges.delete(key);
      const a = this._adj.get(edge.from);
      const b = this._reverseAdj.get(edge.to);
      if (a) a.delete(key);
      if (b) b.delete(key);
    }
  }

  _persist() {
    this.queue.run(this.opts.storageKey, async () => {
      await this.storage.set(this.opts.storageKey, this.toJSON());
    });
  }
}

function edgeKey(from, to, relation) {
  return `${from}|${relation}|${to}`;
}

function normalize(label) {
  return String(label || "")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "_");
}

export default KnowledgeGraph;
