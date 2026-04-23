package com.bookmyshow.entity;

public enum BookingStatus {
    PENDING,     // seats held, waiting for payment
    CONFIRMED,   // payment successful
    FAILED,      // payment failed or booking cancelled
    EXPIRED      // user didn't complete payment in time
}
