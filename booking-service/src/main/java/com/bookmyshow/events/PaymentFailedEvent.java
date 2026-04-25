package com.bookmyshow.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/** Published by PaymentService when a charge is rejected or the gateway is unreachable. */
@Data @NoArgsConstructor @AllArgsConstructor
public class PaymentFailedEvent {
    private Long bookingId;
    private String reason;
    private Instant occurredAt;
}
