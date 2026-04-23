package com.bookmyshow.kafka;

import com.bookmyshow.entity.BookingStatus;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.List;

@Data @NoArgsConstructor @AllArgsConstructor
public class BookingEvent {
    private Long bookingId;
    private Long userId;
    private Long showId;
    private List<String> seatLabels;
    private BookingStatus status;
    private Instant occurredAt;
}
