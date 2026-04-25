package com.bookmyshow.saga;

import com.bookmyshow.entity.Booking;
import com.bookmyshow.entity.BookingStatus;
import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.SeatStatus;
import com.bookmyshow.events.*;
import com.bookmyshow.exception.SeatUnavailableException;
import com.bookmyshow.outbox.OutboxService;
import com.bookmyshow.readmodel.ShowAvailabilityProjector;
import com.bookmyshow.repository.BookingRepository;
import com.bookmyshow.repository.SeatRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * BookingService's side of the choreography saga.
 *
 * The saga flow:
 *
 *     1. startSaga(req)                  [THIS SERVICE]  create Booking(PENDING), HOLD seats,
 *                                                         enqueue SeatsReservedEvent in outbox
 *     2. OutboxPublisher publishes event [THIS SERVICE]  -> Kafka
 *     3. PaymentService listens           [OTHER SVC]     charges card, emits PaymentCharged or PaymentFailed
 *     4a. onPaymentCharged                [THIS SERVICE]  flip Booking to CONFIRMED, seats HELD->BOOKED
 *     4b. onPaymentFailed                 [THIS SERVICE]  flip Booking to FAILED, seats HELD->AVAILABLE (COMPENSATE)
 *     5. NotificationService listens      [OTHER SVC]     sends confirmation email (idempotent)
 *
 * Every handler below is IDEMPOTENT: re-processing the same event is a no-op thanks to
 * status checks. This is required by at-least-once Kafka delivery.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BookingSagaCoordinator {

    private final SeatRepository seatRepo;
    private final BookingRepository bookingRepo;
    private final OutboxService outbox;
    private final ShowAvailabilityProjector projector;

    /**
     * Step 1: starts the saga. Atomic DB transaction:
     *   - Create Booking in PENDING state
     *   - Flip seats AVAILABLE -> HELD (atomic CAS)
     *   - Insert SeatsReservedEvent into the outbox
     *
     * If any of these fail, everything rolls back - no half-state, no phantom event.
     * The outbox publisher picks up the event after commit and sends it to Kafka.
     */
    @Transactional
    public Long startSaga(Long userId, Long showId, List<Long> seatIds, long amountCents,
                          String idempotencyKey) {
        // Idempotency: if this key was already used, return the existing booking.
        if (idempotencyKey != null) {
            var existing = bookingRepo.findByIdempotencyKey(idempotencyKey);
            if (existing.isPresent()) {
                log.info("Idempotent saga start for key={}, booking={}", idempotencyKey,
                         existing.get().getId());
                return existing.get().getId();
            }
        }

        List<Long> sortedIds = seatIds.stream().sorted().collect(Collectors.toList());

        // Atomic CAS: only flip rows that are still AVAILABLE.
        int changed = seatRepo.atomicUpdateStatusIfCurrent(
                sortedIds, SeatStatus.AVAILABLE, SeatStatus.HELD);
        if (changed != sortedIds.size()) {
            throw new SeatUnavailableException(
                    "Only " + changed + " of " + sortedIds.size() + " seats could be held");
        }

        List<Seat> seats = seatRepo.findAllById(sortedIds);

        Booking booking = Booking.builder()
                .userId(userId)
                .showId(showId)
                .seatIds(sortedIds.stream().map(String::valueOf).collect(Collectors.joining(",")))
                .seatLabels(seats.stream().map(Seat::getSeatLabel).collect(Collectors.joining(",")))
                .status(BookingStatus.PENDING)
                .idempotencyKey(idempotencyKey)
                .createdAt(LocalDateTime.now())
                .build();
        booking = bookingRepo.save(booking);

        SeatsReservedEvent event = new SeatsReservedEvent(
                booking.getId(), userId, showId, sortedIds,
                seats.stream().map(Seat::getSeatLabel).toList(),
                amountCents, idempotencyKey, Instant.now());

        // Critical: this goes into the outbox table in the SAME TRANSACTION as the booking.
        outbox.save(SagaTopics.SEATS_RESERVED, String.valueOf(showId), event);

        log.info("Saga started for booking {}, seats HELD", booking.getId());
        return booking.getId();
    }

    /**
     * Step 4a: payment succeeded. Flip booking to CONFIRMED, seats HELD->BOOKED.
     * Idempotent: if booking is already CONFIRMED/FAILED, we do nothing.
     */
    @KafkaListener(topics = SagaTopics.PAYMENT_CHARGED, groupId = "booking-saga",
                   containerFactory = "sagaKafkaListenerContainerFactory")
    @Transactional
    public void onPaymentCharged(PaymentChargedEvent event) {
        log.info("Saga: payment charged for booking {}", event.getBookingId());

        Booking b = bookingRepo.findById(event.getBookingId()).orElse(null);
        if (b == null) {
            log.warn("Payment charged for unknown booking {} - probably wrong cluster?",
                     event.getBookingId());
            return;
        }
        if (b.getStatus() != BookingStatus.PENDING) {
            log.info("Booking {} already in status {} - idempotent skip",
                     b.getId(), b.getStatus());
            return;
        }

        List<Long> seatIds = Arrays.stream(b.getSeatIds().split(","))
                .map(Long::valueOf).sorted().toList();

        // HELD -> BOOKED (CAS: only seats we hold, nobody else could have flipped them)
        seatRepo.atomicUpdateStatusIfCurrent(seatIds, SeatStatus.HELD, SeatStatus.BOOKED);

        b.setStatus(BookingStatus.CONFIRMED);
        bookingRepo.save(b);

        // Recompute read model immediately (Kafka consumer will also recompute, but
        // doing it here makes the HTTP read consistent sooner).
        projector.recomputeFor(b.getShowId());
    }

    /**
     * Step 4b (COMPENSATION): payment failed. Release the seats we were holding.
     *
     * This is the compensating action for step 1. It is NOT a rollback - the booking
     * row still exists (now as FAILED) so we have an audit trail of the attempt.
     */
    @KafkaListener(topics = SagaTopics.PAYMENT_FAILED, groupId = "booking-saga",
                   containerFactory = "sagaKafkaListenerContainerFactory")
    @Transactional
    public void onPaymentFailed(PaymentFailedEvent event) {
        log.info("Saga: compensating for booking {} (reason: {})",
                 event.getBookingId(), event.getReason());

        Booking b = bookingRepo.findById(event.getBookingId()).orElse(null);
        if (b == null) return;
        if (b.getStatus() != BookingStatus.PENDING) {
            log.info("Booking {} already in status {} - idempotent compensation skip",
                     b.getId(), b.getStatus());
            return;
        }

        List<Long> seatIds = Arrays.stream(b.getSeatIds().split(","))
                .map(Long::valueOf).sorted().toList();

        // Release the held seats: HELD -> AVAILABLE
        seatRepo.atomicUpdateStatusIfCurrent(seatIds, SeatStatus.HELD, SeatStatus.AVAILABLE);

        b.setStatus(BookingStatus.FAILED);
        bookingRepo.save(b);

        projector.recomputeFor(b.getShowId());
        log.info("Saga: booking {} compensated, seats released", b.getId());
    }

    /**
     * Compensation fan-out: if a later saga step (e.g. notification) fails permanently,
     * publishing a BookingCompensateEvent triggers refund (PaymentService) + seat release (here).
     */
    @KafkaListener(topics = SagaTopics.BOOKING_COMPENSATE, groupId = "booking-saga",
                   containerFactory = "sagaKafkaListenerContainerFactory")
    @Transactional
    public void onCompensate(BookingCompensateEvent event) {
        Booking b = bookingRepo.findById(event.getBookingId()).orElse(null);
        if (b == null) return;
        if (b.getStatus() == BookingStatus.FAILED) return;  // already compensated

        List<Long> seatIds = Arrays.stream(b.getSeatIds().split(","))
                .map(Long::valueOf).sorted().toList();
        seatRepo.atomicUpdateStatusIfCurrent(seatIds, SeatStatus.BOOKED, SeatStatus.AVAILABLE);
        seatRepo.atomicUpdateStatusIfCurrent(seatIds, SeatStatus.HELD, SeatStatus.AVAILABLE);

        b.setStatus(BookingStatus.FAILED);
        bookingRepo.save(b);
        projector.recomputeFor(b.getShowId());
        log.info("Saga compensation fan-out completed for booking {}", b.getId());
    }
}
