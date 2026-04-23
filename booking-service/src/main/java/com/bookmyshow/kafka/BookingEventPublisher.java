package com.bookmyshow.kafka;

import com.bookmyshow.config.KafkaTopicsConfig;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

/**
 * Produces events only AFTER the booking transaction commits. The caller
 * wires this into TransactionSynchronization#afterCommit so we never publish
 * a "confirmed" message for a booking the DB rolled back.
 *
 * Partition key = showId, so all events for the same show land on the same
 * partition and a downstream consumer can maintain per-show state without races.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class BookingEventPublisher {

    private final KafkaTemplate<String, Object> kafka;

    public void publishConfirmed(BookingEvent event) {
        kafka.send(KafkaTopicsConfig.BOOKING_CONFIRMED, String.valueOf(event.getShowId()), event)
             .whenComplete((r, ex) -> {
                 if (ex != null) log.error("Failed to publish confirmed event {}", event, ex);
             });
    }

    public void publishFailed(BookingEvent event) {
        kafka.send(KafkaTopicsConfig.BOOKING_FAILED, String.valueOf(event.getShowId()), event);
    }
}
