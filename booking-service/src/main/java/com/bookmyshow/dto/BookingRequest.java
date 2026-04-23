package com.bookmyshow.dto;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.List;

@Data
public class BookingRequest {
    @NotNull  private Long userId;
    @NotNull  private Long showId;
    @NotEmpty private List<Long> seatIds;

    /** Optional: which locking strategy to use. Controller defaults it if missing. */
    private String strategy;
}
