package com.bookmyshow.outbox;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

/**
 * Outbox pattern row.
 *
 * The service writes business data AND this row in ONE database transaction.
 * A separate worker polls rows where published=false and sends them to Kafka.
 * Because the INSERT is part of the same tx as the business change, it's impossible
 * to "publish without the business state being durable" or vice versa.
 *
 * This avoids the dual-write problem (DB + Kafka) without needing 2PC.
 */
@Entity
@Table(name = "outbox_event", indexes = {
        @Index(name = "idx_outbox_unpublished", columnList = "published,createdAt")
})
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class OutboxEvent {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Kafka topic to publish to. */
    private String topic;

    /** Partition key (e.g. showId or bookingId); preserves ordering for the same key. */
    @Column(name = "partition_key")
    private String partitionKey;

    /** JSON-serialized payload. */
    @Column(columnDefinition = "TEXT")
    private String payload;

    /** Fully qualified class name of the payload — helps the worker deserialize/log. */
    @Column(name = "payload_type")
    private String payloadType;

    private boolean published;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "published_at")
    private Instant publishedAt;

    @Column(name = "attempts")
    private int attempts;

    @Column(name = "last_error", length = 500)
    private String lastError;
}
