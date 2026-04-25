package com.bookmyshow.payment.service;

import com.bookmyshow.payment.entity.Payment;
import com.bookmyshow.payment.events.BookingCompensateEvent;
import com.bookmyshow.payment.events.PaymentChargedEvent;
import com.bookmyshow.payment.events.PaymentFailedEvent;
import com.bookmyshow.payment.events.SeatsReservedEvent;
import com.bookmyshow.payment.repository.PaymentRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

/**
 * The PaymentService's role in the saga.
 *
 *   IN:  saga.seats.reserved   -> charge card
 *   OUT: saga.payment.charged  (success) | saga.payment.failed (failure)
 *
 *   IN:  saga.booking.compensate -> refund card (if we previously charged it)
 *
 * Idempotency:
 *   - DB unique constraint on Payment.bookingId means a duplicate event cannot
 *     create a second Payment row. We catch the constraint violation and treat
 *     the event as already processed.
 *   - The compensation handler is also idempotent (status check before refunding).
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class PaymentSagaHandler {

    private static final String TOPIC_PAYMENT_CHARGED  = "saga.payment.charged";
    private static final String TOPIC_PAYMENT_FAILED   = "saga.payment.failed";

    private final PaymentRepository paymentRepo;
    private final PaymentGateway gateway;
    private final KafkaTemplate<String, Object> kafka;

    @KafkaListener(topics = "saga.seats.reserved", groupId = "payment-service")
    public void onSeatsReserved(SeatsReservedEvent event) {
        log.info("Charging for booking {} amount {}", event.getBookingId(), event.getAmountCents());

        // Fast-path idempotency: if we already processed this booking, skip.
        if (paymentRepo.existsByBookingId(event.getBookingId())) {
            log.info("Payment for booking {} already exists, skipping (idempotent)",
                     event.getBookingId());
            return;
        }

        try {
            String txnRef = gateway.charge(event.getUserId(), event.getAmountCents());
            persistAndPublishSuccess(event, txnRef);
        } catch (Exception e) {
            log.warn("Charge failed for booking {}: {}", event.getBookingId(), e.getMessage());
            persistAndPublishFailure(event, e.getMessage());
        }
    }

    /**
     * Charge succeeded. Persist Payment and publish PaymentChargedEvent IN ONE TRANSACTION
     * via try/catch fallback for the unique constraint.
     *
     * Note: a strict outbox would write the event to a payment-side outbox table here
     * for guaranteed delivery. We keep this teaching project simpler by publishing
     * directly and relying on retries; see booking-service for the full outbox pattern.
     */
    @Transactional
    public void persistAndPublishSuccess(SeatsReservedEvent event, String txnRef) {
        Payment p = Payment.builder()
                .bookingId(event.getBookingId())
                .userId(event.getUserId())
                .amountCents(event.getAmountCents())
                .transactionRef(txnRef)
                .status(Payment.Status.CHARGED)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
        try {
            p = paymentRepo.save(p);
        } catch (DataIntegrityViolationException dup) {
            log.info("Race: another worker already saved payment for booking {}",
                     event.getBookingId());
            return;
        }
        kafka.send(TOPIC_PAYMENT_CHARGED, String.valueOf(event.getBookingId()),
                new PaymentChargedEvent(event.getBookingId(), p.getId(), txnRef,
                                        event.getAmountCents(), Instant.now()));
    }

    @Transactional
    public void persistAndPublishFailure(SeatsReservedEvent event, String reason) {
        Payment p = Payment.builder()
                .bookingId(event.getBookingId())
                .userId(event.getUserId())
                .amountCents(event.getAmountCents())
                .status(Payment.Status.FAILED)
                .reason(reason)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
        try {
            paymentRepo.save(p);
        } catch (DataIntegrityViolationException dup) {
            // Already recorded the outcome; still publish the failure for the saga to act.
        }
        kafka.send(TOPIC_PAYMENT_FAILED, String.valueOf(event.getBookingId()),
                new PaymentFailedEvent(event.getBookingId(), reason, Instant.now()));
    }

    /**
     * Compensation: refund a previously successful payment. Idempotent.
     * Guards: only act on CHARGED rows; a second compensate finds REFUNDED and skips.
     */
    @KafkaListener(topics = "saga.booking.compensate", groupId = "payment-service")
    @Transactional
    public void onCompensate(BookingCompensateEvent event) {
        Payment p = paymentRepo.findByBookingId(event.getBookingId()).orElse(null);
        if (p == null) {
            log.info("No payment to refund for booking {}", event.getBookingId());
            return;
        }
        if (p.getStatus() != Payment.Status.CHARGED) {
            log.info("Payment {} status={}, idempotent skip", p.getId(), p.getStatus());
            return;
        }
        // In real life: gateway.refund(p.getTransactionRef()).
        p.setStatus(Payment.Status.REFUNDED);
        p.setUpdatedAt(Instant.now());
        paymentRepo.save(p);
        log.info("Refunded payment {} for booking {}", p.getId(), event.getBookingId());
    }
}
