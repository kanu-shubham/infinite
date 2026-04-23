package com.bookmyshow.entity;

import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;

/**
 * A seat in a show. This is the contested resource: two users trying to book
 * the same (show, seat) combination is the core concurrency problem.
 *
 * Optimistic locking via @Version is a compare-and-swap: on UPDATE, Hibernate
 * adds `WHERE version = :oldVersion` and bumps the version. If two transactions
 * both read version=0 and both try to update, one of them gets 0 rows affected
 * and Hibernate throws ObjectOptimisticLockingFailureException. No DB-level
 * row lock is taken during the read, so it scales well under low contention.
 */
@Entity
@Table(name = "seat",
       uniqueConstraints = @UniqueConstraint(columnNames = {"show_id", "seat_label"}))
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Seat implements Serializable {
    @Id
    private Long id;

    @Column(name = "show_id")
    private Long showId;

    @Column(name = "seat_label")
    private String seatLabel;

    @Enumerated(EnumType.STRING)
    private SeatStatus status;

    @Version
    private Long version;
}
