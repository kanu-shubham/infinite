package com.bookmyshow.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/** Published by PaymentService after a successful charge. */
@Data @NoArgsConstructor @AllArgsConstructor
public class PaymentChargedEvent {
    private Long bookingId;
    private Long paymentId;
    private String transactionRef;
    private long amountCents;
    private Instant occurredAt;
}
