package com.bookmyshow.events;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;

/** Published by NotificationService after a confirmation has been delivered. */
@Data @NoArgsConstructor @AllArgsConstructor
public class NotificationSentEvent {
    private Long bookingId;
    private String channel;
    private Instant occurredAt;
}
