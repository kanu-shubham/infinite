package com.bookmyshow.dto;

import com.bookmyshow.entity.BookingStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class BookingResponse {
    private Long bookingId;
    private Long showId;
    private List<String> seatLabels;
    private BookingStatus status;
    private String message;
}
