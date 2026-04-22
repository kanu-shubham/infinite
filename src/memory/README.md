# Memory Architectures

A zero-dependency, production-grade memory library that covers four
architectures behind one orchestrator:

1. **Short-term contextual memory (STM)** — bounded sliding-window buffer
   of recent turns, ranked by recency × salience.
2. **Long-term episodic memory (LTM)** — persistent store of discrete
   experiences with similarity recall, decay (Ebbinghaus), and
   reinforcement on access.
3. **Knowledge graph augmentation (KG)** — weighted nodes and edges,
   neighbor queries, weighted BFS traversal, heuristic extraction from
   LTM episodes.
4. **Adaptive retrieval (AR)** — fuses STM/LTM/KG/recency signals behind a
   learned weighting. An epsilon-greedy bandit tunes the weights from
   feedback (`reward ∈ [-1, 1]`) on served results.

The library is framework-agnostic; a React integration layer is shipped
alongside for the hotels-app demo.

## Quickstart

```js
import {
  createMemoryManager,
  LocalStorageAdapter,
  createTransformerEmbedder,
} from "./memory";

const manager = createMemoryManager({
  storage: new LocalStorageAdapter({ namespace: "my-app" }),
  // Any encoder returning Float32Array | number[] plugs in here:
  // embedder: createTransformerEmbedder({ dim: 384, encodeFn: myEncoder }),
});

await manager.init();

// Write: STM always, LTM when salient or `persist: true`, KG augmented
// via the LTM event pipeline.
await manager.observe({
  text: "search for seaside hotel with pool",
  salience: 2,
  persist: true,
  subjectId: "query:seaside",
});

// Read: adaptive retrieval across the four sources.
const { results, weights, latencyMs } = await manager.query(
  "seaside pool hotel",
  { k: 8 }
);

// Feedback closes the adaptive loop.
manager.feedback({ resultId: results[0].id, reward: 1 });
```

## React

```jsx
import {
  MemoryProvider,
  createMemoryManager,
  useRecall,
  useShortTermMemory,
  useMemoryMetrics,
} from "./memory";

function App() {
  const manager = useMemo(() => createMemoryManager(), []);
  return (
    <MemoryProvider manager={manager}>
      <Dashboard />
    </MemoryProvider>
  );
}

function Dashboard() {
  const { items } = useShortTermMemory();
  const { results, feedback } = useRecall("pool spa", { k: 5 });
  const metrics = useMemoryMetrics();
  // ...
}
```

## Architecture

```
┌───────────────── MemoryManager ─────────────────┐
│                                                 │
│   observe(event) ─▶  STM.add ─▶ (salient?) ─▶   │
│                                LTM.remember ──▶ │ emits → KG.augmenter
│                                                 │
│   query(text) ───▶  AdaptiveRetriever           │
│                     ├─ STM.rank                 │
│                     ├─ LTM.recall               │
│                     ├─ KG.traverse              │
│                     └─ fuse via bandit weights  │
│                                                 │
│   feedback({id, reward}) ─▶ bandit.update(arm)  │
│                                                 │
└─────────────────────────────────────────────────┘
```

Every subsystem also works standalone — `ShortTermMemory`,
`LongTermMemory`, `KnowledgeGraph`, and `AdaptiveRetriever` are exported
from the barrel.

## Pluggable interfaces

### Embedder

```ts
{
  name: string,
  dim: number | null,
  encode(text): Promise<Map<string,number> | Float32Array>,
  sim(a, b): number,
  batch?(texts): Promise<Vector[]>
}
```

Defaults to bag-of-words cosine. Swap in any encoder (transformers.js,
ONNX, remote API) via `createTransformerEmbedder({ dim, encodeFn })`.

### StorageAdapter

```ts
{
  open(): Promise<void>,
  get(key): Promise<any | null>,
  set(key, value): Promise<void>,
  del(key): Promise<void>,
  list(prefix?): Promise<string[]>,
  version(): Promise<number>,
  setVersion(n): Promise<void>,
  close(): Promise<void>
}
```

Built-in adapters: `MemoryAdapter`, `LocalStorageAdapter`. Schema
migrations live in `storage/migrations.js`; register new ones with
`registerMigration({ from, to, up(adapter) })`.

## Events

Subscribe via `manager.on(event, fn)`:

- `stm:add`, `stm:evict`, `stm:boost`, `stm:clear`
- `ltm:remember`, `ltm:recall`, `ltm:forget`
- `kg:upsert` (`{ kind: "node" | "edge" }`)
- `retrieval:done`, `feedback:applied`
- `decay:run`, `storage:quotaExceeded`
- `manager:ready`, `manager:disposed`

`createMetricsCollector(emitter)` aggregates these into rolling
percentiles, counters, and current sizes — wired by default through
`manager.metrics`.

## Demo (hotels app)

The library powers a `MemoryPanel` drawer on the hotels listing page:

- Recent — STM window (every committed filter change).
- Suggested — live `useRecall` over the current search query.
- Graph — KG neighbors of entities mentioned in the search.
- Metrics — counts, recall p50/p95, bandit weights.

Clicking **Save** on a hotel card writes an LTM episode and pushes
`reward: +1` into the retriever; **Hide** forgets the episode and pushes
`reward: -1`. Over time the bandit reweights STM/LTM/KG/recency for the
user.

## Testing

```bash
npm test -- --watchAll=false
```

## Trade-offs

- Lexical bag-of-words is the baseline; quality improves by swapping the
  embedder, not by rewriting modules.
- `LocalStorageAdapter` caps at ~5 MB and surfaces
  `storage:quotaExceeded` — prefer IndexedDB beyond ~1k episodes.
- The bandit warms up on fixed weights for the first `warmupQueries`
  calls to avoid cold-start oscillation.
- The heuristic KG extractor is intentionally simple. Register a richer
  extractor via `createMemoryManager({ extractor })` to upgrade without
  touching the graph module.
