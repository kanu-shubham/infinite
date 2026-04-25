package com.bookmyshow.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/**
 * Emitted to trigger compensating actions across the saga when a later step fails.
 * Subscribers (PaymentService for refund, BookingService for seat release) act idempotently.
 */
@Data @NoArgsConstructor @AllArgsConstructor
public class BookingCompensateEvent {
    private Long bookingId;
    private String reason;
    private Instant occurredAt;
}
