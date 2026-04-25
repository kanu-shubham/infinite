package com.bookmyshow.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.List;

/** Published by BookingService when seats are HELD and payment should be attempted. */
@Data @NoArgsConstructor @AllArgsConstructor
public class SeatsReservedEvent {
    private Long bookingId;
    private Long userId;
    private Long showId;
    private List<Long> seatIds;
    private List<String> seatLabels;
    private long amountCents;
    private String idempotencyKey;
    private Instant occurredAt;
}
