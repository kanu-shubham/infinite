package com.bookmyshow.notification.service;

import com.bookmyshow.notification.entity.Notification;
import com.bookmyshow.notification.events.NotificationSentEvent;
import com.bookmyshow.notification.events.PaymentChargedEvent;
import com.bookmyshow.notification.repository.NotificationRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

/**
 * NotificationService - the last step of the saga's happy path.
 *
 * Triggered by saga.payment.charged. Sends a (simulated) email and emits
 * saga.notification.sent so the saga's "completed" state is observable on the bus.
 *
 * Idempotent via DB unique constraint on Notification.bookingId.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class NotificationSagaHandler {

    private final NotificationRepository repo;
    private final KafkaTemplate<String, Object> kafka;

    @KafkaListener(topics = "saga.payment.charged", groupId = "notification-service")
    @Transactional
    public void onPaymentCharged(PaymentChargedEvent event) {
        if (repo.existsByBookingId(event.getBookingId())) {
            log.info("Notification for booking {} already sent, skipping (idempotent)",
                     event.getBookingId());
            return;
        }

        // Real impl: call email/SMS provider here. Simulate success.
        log.info("[email] Booking confirmed for bookingId={} txn={}",
                 event.getBookingId(), event.getTransactionRef());

        Notification n = Notification.builder()
                .bookingId(event.getBookingId())
                .channel("email")
                .status(Notification.Status.SENT)
                .sentAt(Instant.now())
                .build();
        try {
            repo.save(n);
        } catch (DataIntegrityViolationException dup) {
            log.info("Race: another worker recorded notification for booking {}",
                     event.getBookingId());
            return;
        }

        kafka.send("saga.notification.sent", String.valueOf(event.getBookingId()),
                new NotificationSentEvent(event.getBookingId(), "email", Instant.now()));
    }
}
