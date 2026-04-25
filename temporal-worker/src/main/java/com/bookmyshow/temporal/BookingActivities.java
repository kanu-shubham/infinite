package com.bookmyshow.temporal;

import io.temporal.activity.ActivityInterface;
import io.temporal.activity.ActivityMethod;

import java.util.List;

/**
 * Activities are the side-effecting work — DB writes, HTTP calls, etc.
 *
 * Workflow code MUST be deterministic (Temporal replays it). Anything non-deterministic
 * (random, time, IO, network) goes inside an activity. Activities are retried by Temporal
 * automatically with configurable backoff.
 */
@ActivityInterface
public interface BookingActivities {

    @ActivityMethod
    Long holdSeats(Long userId, Long showId, List<Long> seatIds);

    @ActivityMethod
    String chargePayment(Long bookingId, Long userId, long amountCents);

    @ActivityMethod
    void confirmBooking(Long bookingId, String txnRef);

    @ActivityMethod
    void releaseSeats(Long bookingId);    // compensator for holdSeats

    @ActivityMethod
    void refundPayment(Long bookingId);   // compensator for chargePayment

    @ActivityMethod
    void sendNotification(Long bookingId);
}
