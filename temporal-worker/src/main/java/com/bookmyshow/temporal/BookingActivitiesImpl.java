package com.bookmyshow.temporal;

import io.temporal.spring.boot.ActivityImpl;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Stub activity implementations.
 *
 * In a real system each method would call out to the corresponding microservice
 * (HTTP REST or gRPC) — for example, BookingService for holdSeats, PaymentService
 * for chargePayment. Activities are normal Spring beans and can inject anything
 * (RestTemplate, JPA repos, Redis, etc.).
 *
 * For this demo we just log + sleep, with random failures so you can watch
 * Temporal retry and compensate.
 */
@Slf4j
@Component
@ActivityImpl(taskQueues = "booking-saga")
public class BookingActivitiesImpl implements BookingActivities {

    @Override
    public Long holdSeats(Long userId, Long showId, List<Long> seatIds) {
        log.info("[activity] holdSeats user={} show={} seats={}", userId, showId, seatIds);
        // Imagine calling BookingService here. Pretend bookingId=42.
        return 42L + ThreadLocalRandom.current().nextLong(1000);
    }

    @Override
    public String chargePayment(Long bookingId, Long userId, long amountCents) {
        log.info("[activity] chargePayment booking={} amount={}", bookingId, amountCents);
        // 20% chance of transient failure -> Temporal retries automatically.
        if (ThreadLocalRandom.current().nextInt(100) < 20) {
            throw new RuntimeException("transient gateway error");
        }
        return "txn_" + System.nanoTime();
    }

    @Override
    public void confirmBooking(Long bookingId, String txnRef) {
        log.info("[activity] confirmBooking booking={} txn={}", bookingId, txnRef);
    }

    @Override
    public void releaseSeats(Long bookingId) {
        log.info("[activity COMPENSATE] releaseSeats booking={}", bookingId);
    }

    @Override
    public void refundPayment(Long bookingId) {
        log.info("[activity COMPENSATE] refundPayment booking={}", bookingId);
    }

    @Override
    public void sendNotification(Long bookingId) {
        log.info("[activity] sendNotification booking={}", bookingId);
    }
}
