# BookMyShow Backend — concurrency, saga, outbox, CQRS, circuit breakers, workflow engine

A teaching repo that grew from "single Spring Boot service that handles seat
contention" to a small but realistic distributed system. Every concept is
runnable and verifiable end-to-end.

## What's in here

| Module | What it teaches |
|---|---|
| `booking-service/` | Three concurrent locking strategies, Redis cache, distributed lock, async payment, **outbox pattern**, **read model / CQRS projector**, **saga coordinator**, **intentionally broken `synchronized` demo**, 100-thread load test |
| `payment-service/` | Own DB, **Resilience4j circuit breaker + retry**, idempotent Kafka consumer, compensating refund handler |
| `notification-service/` | Own DB, idempotent saga step, last hop of the choreography |
| `temporal-worker/` | Same saga rewritten as a **Temporal workflow** — durable execution, automatic retries, surviving worker crashes |
| `gatling-loadtest/` | **1000 rps for 10 minutes** load profile with assertions on p95/p99 |
| `nginx/` | Round-robin reverse proxy for the multi-instance demo |
| `docker-compose.yml` | Redis + Kafka + 3 Postgres instances + Temporal + nginx |

## Quick run (single service, zero infrastructure)

```bash
cd booking-service
mvn spring-boot:run
# Open http://localhost:8080/api/movies
```

Uses embedded H2; Redis/Kafka features are skipped automatically when those services are unreachable.

## Full stack (docker compose)

```bash
docker compose up -d redis zookeeper kafka postgres-booking postgres-payment postgres-notification

# Three terminals (one each):
( cd booking-service      && mvn spring-boot:run )
( cd payment-service      && mvn spring-boot:run )
( cd notification-service && mvn spring-boot:run )
```

Service ports: booking 8080, payment 8081, notification 8082.

---

## DEMO 1 — Three locking strategies + the broken `synchronized` demo

Hammer the same seat with 100 threads, see exactly one win:

```bash
curl -X POST 'localhost:8080/api/loadtest?strategy=optimistic&threads=100&seatId=1'
curl -X POST 'localhost:8080/api/loadtest?strategy=pessimistic&threads=100&seatId=2'
curl -X POST 'localhost:8080/api/loadtest?strategy=distributed&threads=100&seatId=3'

# All three return: success=1, conflict=99
```

Now the broken one. **Single instance**: `synchronized` works inside one JVM, so this still says `success=1` (passing for the wrong reason). **Two instances behind nginx**: it breaks.

```bash
# Single instance — passes (incorrectly)
curl -X POST 'localhost:8080/api/loadtest?strategy=broken-synchronized&threads=100&seatId=4'

# (After scaling — see DEMO 6 — repeat against http://localhost:8090)
# It will return success > 1. The seat is sold twice. That is the bug.
```

**Where to read:**
- `booking-service/src/main/java/com/bookmyshow/service/BookingService.java` — three correct strategies
- `booking-service/src/main/java/com/bookmyshow/broken/SynchronizedBookingService.java` — the broken one

---

## DEMO 2 — Saga (choreography) with all three services

Start the three services. Then:

```bash
curl -X POST localhost:8080/api/saga/bookings \
     -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: saga-test-1' \
     -d '{"userId":1,"showId":1,"seatIds":[5,6]}'
# 202 Accepted, returns {"bookingId":1, "status":"PENDING", ...}
```

What happens behind the curtain:

1. `BookingService` runs a single DB transaction: HOLD seats + create Booking(PENDING) + insert SeatsReservedEvent into the **outbox table**.
2. `OutboxPublisher` (scheduled, 500ms) reads the outbox, sends the event to Kafka topic `saga.seats.reserved`, marks row `published=true`.
3. `PaymentService` consumes it, charges the (simulated) gateway, persists `Payment` row, publishes `saga.payment.charged` (or `saga.payment.failed`).
4. `BookingService.onPaymentCharged` flips the booking to CONFIRMED and seats HELD→BOOKED.
5. `NotificationService` consumes `saga.payment.charged`, "sends" the email, publishes `saga.notification.sent`.

Poll the result:

```bash
curl localhost:8080/api/saga/bookings/1
# {"bookingId":1, "status":"CONFIRMED", "seatLabels":"A5,A6"}

curl localhost:8081/payments/booking/1
# {"id":..., "bookingId":1, "status":"CHARGED", "transactionRef":"txn_..."}
```

**Test the compensation path** by forcing payment failure:

```bash
PAYMENT_FAIL_MODE=always mvn -pl payment-service spring-boot:run

# Then submit a saga
curl -X POST localhost:8080/api/saga/bookings \
     -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: saga-fail-1' \
     -d '{"userId":2,"showId":1,"seatIds":[7]}'

# Booking goes PENDING -> FAILED, seats released back to AVAILABLE.
# Watch the BookingService log: "Saga: compensating for booking N (reason: ...)".
```

**Crash mid-charge**: kill `payment-service` while a saga is in-flight, restart it. Kafka redelivers the un-acked event; the DB unique constraint on `payment.booking_id` makes the retry idempotent — exactly one Payment row regardless of how many times the consumer crashes.

**Where to read:**
- `booking-service/src/main/java/com/bookmyshow/saga/BookingSagaCoordinator.java`
- `payment-service/src/main/java/com/bookmyshow/payment/service/PaymentSagaHandler.java`
- `notification-service/src/main/java/com/bookmyshow/notification/service/NotificationSagaHandler.java`

---

## DEMO 3 — Outbox pattern survives Kafka downtime

```bash
# Submit a saga while everything's running, then kill Kafka:
docker compose stop kafka

# Submit MORE sagas - they succeed (Booking row + outbox row are committed in DB);
# the OutboxPublisher logs warnings about send failures, attempts++ on the row.
for i in 1 2 3 4 5; do
  curl -s -X POST localhost:8080/api/saga/bookings \
       -H 'Content-Type: application/json' \
       -H "Idempotency-Key: kafka-down-$i" \
       -d "{\"userId\":$i,\"showId\":1,\"seatIds\":[$((i+10))]}" &
done

# Bring Kafka back:
docker compose start kafka

# Within ~30 seconds, the outbox drains and PaymentService picks up all five events.
# Look for "Outbox: publishing N events" in booking-service logs.
```

Inspect the table:

```bash
# H2 console at http://localhost:8080/h2-console (jdbc:h2:mem:bookings)
# SELECT id, topic, published, attempts, last_error FROM outbox_event ORDER BY id DESC;
```

**Where to read:**
- `booking-service/src/main/java/com/bookmyshow/outbox/OutboxEvent.java`
- `booking-service/src/main/java/com/bookmyshow/outbox/OutboxPublisher.java`

---

## DEMO 4 — CQRS read model

Compare write-model query (`/api/shows/{id}/seats`, returns 20 seat rows) vs. read-model query (`/api/shows/{id}/availability`, returns one denormalized row):

```bash
# Trigger the projector by booking through the saga
curl -X POST localhost:8080/api/saga/bookings -H 'Content-Type: application/json' \
     -d '{"userId":1,"showId":1,"seatIds":[8]}'

# Read model — O(1)
curl localhost:8080/api/shows/1/availability
# {"showId":1,"availableCount":17,"heldCount":0,"bookedCount":3,"totalCount":20,...}
```

Run the Gatling test against both endpoints to see the latency difference:

```bash
mvn -pl gatling-loadtest gatling:test \
    -Dgatling.simulationClass=simulations.BookingSimulation \
    -DbaseUrl=http://localhost:8080
```

**Where to read:**
- `booking-service/src/main/java/com/bookmyshow/readmodel/ShowAvailability.java`
- `booking-service/src/main/java/com/bookmyshow/readmodel/ShowAvailabilityProjector.java`

---

## DEMO 5 — Circuit breaker watch in real time

```bash
# Force the gateway to always fail and start payment-service
PAYMENT_FAIL_MODE=always mvn -pl payment-service spring-boot:run

# Drive load
for i in $(seq 1 20); do
  curl -s -X POST localhost:8080/api/saga/bookings \
       -H "Idempotency-Key: cb-$i" \
       -H 'Content-Type: application/json' \
       -d "{\"userId\":$i,\"showId\":1,\"seatIds\":[$i]}" &
done

# Watch the breaker trip:
watch -n1 curl -s localhost:8081/payments/breaker
# state: CLOSED -> OPEN after 5+ failures at >50% rate.
# Subsequent calls fail fast (no 30s thread hangs); after 5s it goes HALF_OPEN, probes, recovers.
```

You can also see breaker events at `http://localhost:8081/actuator/circuitbreakerevents`.

**Where to read:**
- `payment-service/src/main/resources/application.yml` (resilience4j config)
- `payment-service/src/main/java/com/bookmyshow/payment/service/PaymentGateway.java`

---

## DEMO 6 — Two booking-service instances behind nginx, prove `synchronized` fails

The cleanest way is to run two Java processes on different ports and let nginx round-robin:

```bash
# Terminal A
SERVER_PORT=8080 mvn -pl booking-service spring-boot:run

# Terminal B
SERVER_PORT=8081 mvn -pl booking-service spring-boot:run

# Update nginx upstream to point at host.docker.internal:8080 and 8081, or run both as containers.
docker compose up -d nginx

# Run the broken strategy through nginx (port 8090)
curl -X POST 'localhost:8090/api/loadtest?strategy=broken-synchronized&threads=100&seatId=15'
# success > 1   <-- the bug.

# Run the correct ones — still success=1
curl -X POST 'localhost:8090/api/loadtest?strategy=optimistic&threads=100&seatId=16'
curl -X POST 'localhost:8090/api/loadtest?strategy=distributed&threads=100&seatId=17'
```

The point: in-JVM `synchronized` is invisible across JVMs. DB-level locks (optimistic CAS, pessimistic `FOR UPDATE`) and Redis distributed locks all still serialize correctly because their state lives in shared infrastructure.

---

## DEMO 7 — Temporal workflow (durable execution)

```bash
docker compose up -d temporal temporal-ui postgres-temporal
( cd temporal-worker && mvn spring-boot:run )

# Submit
curl -X POST localhost:8083/workflows/bookings \
     -H 'Content-Type: application/json' \
     -d '{"userId":1,"showId":1,"seatIds":[20],"amountCents":500}'
# {"workflowId":"booking-..."}

# Get the result (blocks until complete)
curl localhost:8083/workflows/bookings/<workflowId>/result

# Open the Temporal UI: http://localhost:8088
# You'll see every activity, every retry, every event in the workflow's history.
```

**Crash demo**: `Ctrl-C` the temporal-worker mid-workflow, restart it. Temporal replays the persisted history and the workflow resumes from the next un-completed activity. Compare to the manual outbox + saga code — Temporal removes a category of bugs.

**Where to read:**
- `temporal-worker/src/main/java/com/bookmyshow/temporal/BookingWorkflowImpl.java` (the entire saga as plain sequential Java)

---

## DEMO 8 — Gatling at 1000 rps

```bash
# Default: 1000 rps for 10 minutes against localhost:8080.
mvn -pl gatling-loadtest gatling:test \
    -Dgatling.simulationClass=simulations.BookingSimulation

# HTML report path printed at the end. Open it.
# Assertions enforce: p95<500ms, p99<2s, errors<1%.
```

Tunables: `-Drps=2000 -Dduration=120 -DbaseUrl=http://localhost:8090` etc.

**Where to read:**
- `gatling-loadtest/src/test/java/simulations/BookingSimulation.java`

---

## Concept map

| Concept | File |
|---|---|
| Race condition + 3 fixes | `booking-service/.../service/BookingService.java` |
| Optimistic lock (`@Version` + CAS) | `.../entity/Seat.java`, `SeatRepository.atomicUpdateStatusIfCurrent` |
| Pessimistic lock (`SELECT ... FOR UPDATE`) | `SeatRepository.findAllByIdForUpdate` |
| Redis distributed lock + deadlock-safe ordering | `lock/DistributedLockService.java` |
| `synchronized` only works inside one JVM | `broken/SynchronizedBookingService.java` |
| Thread pools, `@Async`, `CallerRunsPolicy` | `config/AsyncConfig.java` |
| `CompletableFuture` from a controller | `controller/BookingController.bookAsync` |
| Cache-aside (`@Cacheable`, per-cache TTL) | `config/RedisConfig.java`, `service/MovieService.java` |
| Lua-scripted rate limiter (atomic INCR+EXPIRE) | `ratelimit/RedisRateLimiter.java` |
| Idempotency keys (Redis SETNX + DB unique index) | `service/IdempotencyService.java` |
| Transactional outbox | `outbox/OutboxEvent.java`, `OutboxPublisher.java` |
| Saga (choreography) | `saga/BookingSagaCoordinator.java` + `payment-service` + `notification-service` |
| Compensating transactions | `BookingSagaCoordinator.onPaymentFailed`, `PaymentSagaHandler.onCompensate` |
| CQRS read model | `readmodel/ShowAvailabilityProjector.java` |
| Circuit breaker + retry + time limiter | `payment-service/.../PaymentGateway.java` + `application.yml` |
| Durable workflow engine | `temporal-worker/...` |
| Realistic load test | `gatling-loadtest/...` |
| Kafka per-partition ordering (`concurrency=N`) | `kafka/NotificationConsumer.java` |

## Suggested learning order

1. Run demo 1 — convince yourself `success=1` for each correct strategy.
2. Read `BookingService.java` end to end.
3. Run demo 2 — start a saga, watch the three services log in turn.
4. Run demo 3 — kill Kafka mid-saga; watch the outbox drain when it comes back.
5. Run demo 5 — trip the circuit breaker.
6. Run demo 6 — see the broken synchronized version oversell a seat.
7. Run demo 7 — kill Temporal worker mid-workflow; watch resume.
8. Run demo 8 — find your bottleneck under 1000 rps.

By demo 8 every concept on the cheat sheet stops being abstract.
