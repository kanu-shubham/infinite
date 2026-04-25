package com.bookmyshow.readmodel;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

/**
 * Read-model projection of seat availability per show.
 *
 * This is DENORMALIZED — the source of truth is the `seat` table. A Kafka consumer
 * (ShowAvailabilityProjector) keeps this row in sync. Reads for "how many seats
 * are left?" hit this row (O(1)) instead of doing COUNT(*) over the seat table.
 *
 * This is CQRS in miniature: the write model is `seat`, the read model is this.
 */
@Entity
@Table(name = "show_availability")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class ShowAvailability {

    @Id
    @Column(name = "show_id")
    private Long showId;

    @Column(name = "available_count")
    private int availableCount;

    @Column(name = "held_count")
    private int heldCount;

    @Column(name = "booked_count")
    private int bookedCount;

    @Column(name = "total_count")
    private int totalCount;

    @Column(name = "updated_at")
    private Instant updatedAt;

    /** Offset/event id to skip duplicate updates — simple at-least-once dedupe. */
    @Column(name = "last_event_id")
    private Long lastEventId;
}
