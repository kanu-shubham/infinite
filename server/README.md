# Netflix Recommendation Bot — Backend

Real-time conversational recommendation service. Spring Boot + Spring AI + Claude
(Anthropic) for chat, OpenAI embeddings, pgvector for retrieval, SSE for streaming.

## How it works end-to-end

```
        client
          │  POST /api/chat/stream  { sessionId, userId, message }
          ▼
   ChatController  ──►  RecommenderAgent
                           │
                           ├─► MessageChatMemoryAdvisor  (last 20 turns)
                           ├─► QuestionAnswerAdvisor     (RAG over pgvector)
                           │       └─► embed(msg) → top-K similar shows → inject
                           ├─► CatalogTools              (function calling)
                           │       ├─ trending()
                           │       ├─ getShow(title)
                           │       ├─ recentWatches()
                           │       └─ addToList(showId)
                           └─► ChatClient.stream()  →  Claude via Anthropic API
                                                       │
                                        Flux<String> tokens
                                                       ▼
                              SSE: event:token … event:done
```

Offline: `CatalogSeeder` loads `data/shows.json`, writes metadata to Postgres,
and calls `CatalogIndexer.index(show)` which embeds each synopsis via
`text-embedding-3-small` and upserts into the `catalog_embeddings` pgvector
table with HNSW + cosine distance.

Online per turn:

1. SSE POST hits `ChatController`.
2. Agent hydrates the user profile (maturity rating, preferred genres, watched IDs).
3. System prompt bakes in personalization rules.
4. `QuestionAnswerAdvisor` embeds the user message, runs a top-8 vector search
   filtered by maturity, and injects matching show chunks into the prompt.
5. `MessageChatMemoryAdvisor` prepends the last 20 conversation turns.
6. Claude decides whether to answer directly or call a tool (e.g. `trending()`).
   If a tool is called, Spring AI executes the Java method, feeds the result back,
   and the loop continues until Claude produces final text.
7. Tokens stream back as SSE `event:token` frames with 15s heartbeats.

## Run it

Prereqs: JDK 21, Maven 3.9+, Docker.

```bash
# 1. Start Postgres + pgvector
docker compose up -d

# 2. Export API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# 3. Boot the service
./mvnw spring-boot:run
```

On first boot the seeder embeds and indexes 15 shows (~3s).

## Try it

```bash
curl -N -X POST http://localhost:8080/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "sessionId": "s-1",
    "userId": "u-alex",
    "message": "I want something dark and twisty like Dark, but shorter episodes"
  }'
```

You'll see:
```
event:token
data:I

event:token
data:'ve got two picks that match...
...
event:done
data:
```

## Production hardening (not in this reference)

| Concern | What to add |
|---|---|
| Memory | Swap `InMemoryChatMemoryRepository` for a Redis-backed repo. |
| Personalization | Compute user embeddings from watch history; rerank LLM output with `α·sim + β·cosine(user, show) + γ·popularity`. |
| Reranking | Cohere Rerank or `bge-reranker-large` between retrieval and LLM. |
| Safety | Prompt-injection detector, PII redactor, topic classifier (see `docs/guardrails.md`). |
| Observability | OTel spans + Langfuse traces around embed / retrieve / LLM / tool calls. |
| Cost | Haiku for routing and short replies, Opus/Sonnet for recommendations. Cache trending/getShow. |
| Eval | Ragas nightly: retrieval recall@8, answer faithfulness, tool-call accuracy. |
| Multi-tenant | Partition vector store by tenant_id; enforce in every filter. |
| Handoff | Add a `needs_human` tool; route to live-agent WebSocket when called (Intercom-style). |

## File map

```
server/
├── pom.xml
├── docker-compose.yml
└── src/main/
    ├── java/com/infinite/netflixbot/
    │   ├── NetflixBotApplication.java
    │   ├── config/WebConfig.java
    │   ├── catalog/
    │   │   ├── Show.java                  — JPA entity
    │   │   ├── ShowRepository.java
    │   │   ├── CatalogIndexer.java        — embeds + upserts into pgvector
    │   │   └── CatalogSeeder.java         — loads shows.json on boot
    │   ├── memory/SessionMemory.java      — ChatMemory bean
    │   ├── user/UserProfile.java
    │   ├── tools/
    │   │   ├── CatalogTools.java          — @Tool methods exposed to LLM
    │   │   └── MyListService.java
    │   └── chat/
    │       ├── RecommenderAgent.java      — the orchestrator
    │       ├── ChatRequest.java
    │       └── ChatController.java        — SSE endpoint
    └── resources/
        ├── application.yml
        └── data/shows.json                — seed catalog
```
