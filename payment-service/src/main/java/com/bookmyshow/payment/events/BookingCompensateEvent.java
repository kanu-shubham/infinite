package com.bookmyshow.payment.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

@Data @NoArgsConstructor @AllArgsConstructor
public class BookingCompensateEvent {
    private Long bookingId;
    private String reason;
    private Instant occurredAt;
}
