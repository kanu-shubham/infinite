package com.bookmyshow.notification.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data @NoArgsConstructor @AllArgsConstructor
public class PaymentChargedEvent {
    private Long bookingId;
    private Long paymentId;
    private String transactionRef;
    private long amountCents;
    private Instant occurredAt;
}
