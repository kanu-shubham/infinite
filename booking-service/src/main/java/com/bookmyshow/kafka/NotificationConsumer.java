package com.bookmyshow.kafka;

import com.bookmyshow.config.KafkaTopicsConfig;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * Reads booking events and "sends" notifications (email / SMS / push).
 *
 * `concurrency = 3` tells Spring-Kafka to run 3 listener threads, one per partition.
 * Each partition is processed single-threaded, which gives us per-show ordering
 * without locking: events for showId=7 are always handled by the same thread.
 */
@Slf4j
@Component
public class NotificationConsumer {

    @KafkaListener(topics = KafkaTopicsConfig.BOOKING_CONFIRMED,
                   groupId = "notification-service",
                   concurrency = "3")
    public void onConfirmed(BookingEvent event) {
        log.info("[notify] Sending confirmation for bookingId={} user={} seats={}",
                 event.getBookingId(), event.getUserId(), event.getSeatLabels());
        // real impl: call email/SMS provider; keep this method idempotent (same
        // event delivered twice is normal in at-least-once Kafka).
    }

    @KafkaListener(topics = KafkaTopicsConfig.BOOKING_FAILED,
                   groupId = "notification-service",
                   concurrency = "3")
    public void onFailed(BookingEvent event) {
        log.info("[notify] Booking {} failed for user {}", event.getBookingId(), event.getUserId());
    }
}
