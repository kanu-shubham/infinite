package com.bookmyshow.outbox;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Periodically drains the outbox table and publishes pending events to Kafka.
 *
 * Safe under Kafka downtime:
 *   - While Kafka is down, KafkaTemplate#send().get() throws; rows stay published=false.
 *   - Worker retries on the next tick. Attempts counter grows; last_error stored for ops.
 *   - When Kafka comes back, the backlog is drained in order.
 *
 * Safe under service restart:
 *   - Rows live in the DB. A crash mid-batch just means the next run republishes a few rows.
 *   - Consumers must be idempotent (our services all dedupe by bookingId).
 *
 * Scaling:
 *   - Single instance: fine for tens of thousands of events/hour.
 *   - Multi-instance: replace the query with `FOR UPDATE SKIP LOCKED` so workers don't
 *     double-publish. Postgres & MySQL 8 both support this.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class OutboxPublisher {

    private final OutboxRepository repo;
    private final KafkaTemplate<String, Object> kafka;
    private final ObjectMapper mapper;

    private static final int BATCH_SIZE = 100;

    @Scheduled(fixedDelayString = "${booking.outbox.poll-delay-ms:500}")
    public void drain() {
        List<OutboxEvent> batch = repo.findUnpublished(PageRequest.of(0, BATCH_SIZE));
        if (batch.isEmpty()) return;

        log.debug("Outbox: publishing {} events", batch.size());
        for (OutboxEvent event : batch) {
            try {
                Object payload = mapper.readValue(event.getPayload(), Class.forName(event.getPayloadType()));
                // .get() blocks with a timeout so a stuck broker cannot hang the worker forever.
                kafka.send(event.getTopic(), event.getPartitionKey(), payload)
                     .get(5, TimeUnit.SECONDS);
                markPublished(event);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                markFailed(event, "interrupted");
                return;
            } catch (ExecutionException | TimeoutException | ClassNotFoundException
                     | com.fasterxml.jackson.core.JsonProcessingException e) {
                markFailed(event, e.getMessage());
            } catch (Exception e) {
                markFailed(event, e.getClass().getSimpleName() + ": " + e.getMessage());
            }
        }
    }

    @Transactional
    protected void markPublished(OutboxEvent event) {
        event.setPublished(true);
        event.setPublishedAt(Instant.now());
        repo.save(event);
    }

    @Transactional
    protected void markFailed(OutboxEvent event, String err) {
        event.setAttempts(event.getAttempts() + 1);
        event.setLastError(err == null ? null : err.substring(0, Math.min(err.length(), 500)));
        repo.save(event);
        if (event.getAttempts() % 10 == 0) {
            log.warn("Outbox event {} failing repeatedly (attempts={}): {}",
                     event.getId(), event.getAttempts(), err);
        }
    }
}
