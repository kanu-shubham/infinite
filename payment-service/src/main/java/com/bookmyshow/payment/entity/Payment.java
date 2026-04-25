package com.bookmyshow.payment.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "payment", indexes = {
        @Index(name = "uk_payment_booking", columnList = "booking_id", unique = true)
})
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Payment {

    public enum Status { CHARGED, FAILED, REFUNDED }

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Unique - one payment record per booking. Enforces idempotency at the DB level. */
    @Column(name = "booking_id", unique = true)
    private Long bookingId;

    private Long userId;

    @Column(name = "amount_cents")
    private long amountCents;

    @Column(name = "transaction_ref")
    private String transactionRef;

    @Enumerated(EnumType.STRING)
    private Status status;

    private String reason;

    @Column(name = "created_at")
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;
}
