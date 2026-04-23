package com.bookmyshow.entity;

public enum SeatStatus {
    AVAILABLE,   // bookable
    HELD,        // reserved during payment window
    BOOKED       // paid and confirmed
}
