# BookMyShow-style Booking Service — a concurrency teaching project

A realistic Spring Boot REST API that lets you **book movie tickets** while
teaching the hard parts of backend engineering hands-on:

| Concept | Where it lives |
|---|---|
| Thread pools, `ExecutorService`, `@Async` | `config/AsyncConfig.java`, `service/PaymentService.java` |
| `CompletableFuture` + async HTTP responses | `controller/BookingController#bookAsync` |
| Pessimistic DB lock (`SELECT ... FOR UPDATE`) | `repository/SeatRepository#findAllByIdForUpdate` |
| Optimistic locking (`@Version` + CAS UPDATE) | `entity/Seat.java`, `SeatRepository#atomicUpdateStatusIfCurrent` |
| Distributed lock (Redis / Redisson MultiLock) | `lock/DistributedLockService.java` |
| Deadlock avoidance (sorted lock order) | `BookingService` & `DistributedLockService` |
| Redis cache-aside (`@Cacheable`, TTLs) | `config/RedisConfig.java`, `MovieService`, `ShowService` |
| Rate limiting (atomic Lua script) | `ratelimit/RedisRateLimiter.java` |
| Idempotency keys (SETNX + DB unique index) | `service/IdempotencyService.java`, `entity/Booking.java` |
| Kafka producer (after-commit only) | `kafka/BookingEventPublisher.java`, `BookingService#registerAfterCommitEvent` |
| Kafka consumer (per-partition ordering) | `kafka/NotificationConsumer.java` |
| 100-thread "thundering herd" load test | `loadtest/LoadTestController.java`, `ConcurrentBookingTest` |

## The scenario

Classic BookMyShow problem: **two users click "book A4" at the same second**.
At most one can win. Get it wrong and you oversell the show.

This project shows three correct ways to solve it, a load-test endpoint to
prove they're correct, and all the surrounding real-world concerns
(caching, rate limiting, async payment, event publishing).

## Run it

Zero-setup mode (in-memory H2, Redis/Kafka disabled — simplest way to poke the
optimistic/pessimistic strategies):

```bash
cd booking-service
./mvnw spring-boot:run
```

Full stack (real Postgres + Redis + Kafka — needed for the distributed lock,
caching, rate limiting, async events):

```bash
docker compose up -d
./mvnw spring-boot:run
```

## Try it

```bash
# Browse catalog (served from Redis cache after first hit)
curl localhost:8080/api/movies
curl localhost:8080/api/movies/1/shows
curl localhost:8080/api/shows/1/seats

# Book - optimistic CAS (default)
curl -X POST localhost:8080/api/bookings \
     -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: abc-123' \
     -d '{"userId":1,"showId":1,"seatIds":[1,2]}'

# Same request again: returns the original booking (idempotency replay)
curl -X POST localhost:8080/api/bookings \
     -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: abc-123' \
     -d '{"userId":1,"showId":1,"seatIds":[1,2]}'

# Switch strategies
curl -X POST 'localhost:8080/api/bookings?strategy=pessimistic' ...
curl -X POST 'localhost:8080/api/bookings?strategy=distributed' ...

# Async booking: payment runs on the payment thread pool, HTTP thread returns immediately
curl -X POST localhost:8080/api/bookings/async \
     -H 'Content-Type: application/json' \
     -d '{"userId":1,"showId":1,"seatIds":[5]}'
```

## The teaching moment — prove the concurrency works

```bash
# 100 threads, all try to book the same seat at once.
# Expected: success=1, conflict=99.
curl -X POST 'localhost:8080/api/loadtest?strategy=optimistic&threads=100&seatId=1'
curl -X POST 'localhost:8080/api/loadtest?strategy=pessimistic&threads=100&seatId=2'
curl -X POST 'localhost:8080/api/loadtest?strategy=distributed&threads=100&seatId=3'
```

If you deliberately break the locking (e.g. remove `@Version` and skip the CAS
update), the load test will show `success > 1` — the demo of an oversold show.

## Strategy cheat sheet

| | Pessimistic | Optimistic | Distributed (Redis) |
|---|---|---|---|
| Mechanism | `SELECT ... FOR UPDATE` | `UPDATE ... WHERE version=N` / CAS | Redisson MultiLock + CAS |
| DB row lock held during | whole tx | microseconds (update only) | short tx inside lock |
| Blocks across JVMs | yes (same DB) | yes (same DB) | **yes, even with multiple app instances** |
| Best when | conflicts frequent, tx short | conflicts rare | lock must span external calls |
| Failure mode | lock timeout | `OptimisticLockingFailureException` | `IllegalStateException` (couldn't acquire) |

## Thread pools at a glance

- **Tomcat worker pool** (`server.tomcat.threads.max=200`) — each inbound HTTP request uses one.
- **`paymentExecutor`** (20 threads, bounded queue, `CallerRunsPolicy`) — slow external calls.
  Returning a `CompletableFuture` from the controller means the Tomcat thread is freed while
  the payment is in flight.
- **`notificationExecutor`** — fire-and-forget.
- **Kafka listener pool** (`concurrency=3`) — one thread per partition, preserving per-show ordering.
- **HikariCP** (`maximum-pool-size=40`) — must be ≥ number of threads that can simultaneously
  hold a JDBC connection, otherwise you get thread-pool-starved-on-DB-connection deadlocks.

## Common pitfalls this project avoids

1. **Deadlock** — both `DistributedLockService` and the pessimistic DB query acquire seats
   in sorted ID order. If you don't do this, two bookers can deadlock on seats `[1,2]` vs `[2,1]`.
2. **Phantom "confirmed" events** — Kafka publish happens in `afterCommit`, never before.
3. **Sync-on-application-restart cache staleness** — JSON serialization + TTLs.
4. **Rate-limit races** — Lua script makes INCR + EXPIRE atomic.
5. **Double-booking via retry** — `Idempotency-Key` header, Redis SETNX, DB unique index.
6. **Slow payment saturating HTTP pool** — `@Async` bounded executor + `CompletableFuture` return.
7. **Crashed holder wedging a lock** — every distributed lock has a TTL (lease).

## Project layout

```
booking-service/
├── src/main/java/com/bookmyshow/
│   ├── config/         thread pools, Redis cache, Kafka topics
│   ├── controller/     REST endpoints
│   ├── dto/            request / response shapes
│   ├── entity/         JPA entities (Movie, Show, Seat, Booking)
│   ├── exception/      typed exceptions + @RestControllerAdvice
│   ├── kafka/          event + producer + listener
│   ├── lock/           Redis distributed lock wrapper
│   ├── loadtest/       100-thread hammer endpoint
│   ├── ratelimit/      Lua-backed rate limiter
│   ├── repository/     Spring Data JPA
│   └── service/        BookingService (3 strategies), MovieService, ShowService, PaymentService
├── src/main/resources/ application.yml, data.sql
├── src/test/java/      ConcurrentBookingTest (100-thread race)
├── docker-compose.yml  Redis + Kafka + Postgres
└── pom.xml
```

## Suggested learning path

1. Start the app, hit `/api/movies` twice, notice the second is served from Redis (`@Cacheable`).
2. POST a booking with `strategy=optimistic`, then try to re-book the same seat — watch the 409.
3. Hit `/api/loadtest?threads=100` and compare `success`/`conflict` counts across strategies.
4. Read `BookingService.java` and explain (to yourself) why `registerAfterCommitEvent` uses
   `TransactionSynchronization` instead of publishing directly.
5. Explain why `DistributedLockService` sorts IDs before acquiring.
6. Scale to 2 app instances behind a reverse proxy — only `strategy=distributed` and the
   DB-backed strategies still give you `success=1`. In-JVM `synchronized` would not.
