package com.bookmyshow.service;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.dto.BookingResponse;
import com.bookmyshow.entity.Booking;
import com.bookmyshow.entity.BookingStatus;
import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.SeatStatus;
import com.bookmyshow.exception.SeatUnavailableException;
import com.bookmyshow.kafka.BookingEvent;
import com.bookmyshow.kafka.BookingEventPublisher;
import com.bookmyshow.lock.DistributedLockService;
import com.bookmyshow.repository.BookingRepository;
import com.bookmyshow.repository.SeatRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Three concurrent booking strategies, all exposed so you can compare them:
 *
 *   PESSIMISTIC  - SELECT ... FOR UPDATE. Simple and correct, but serializes
 *                  contenders on the DB. Choose when conflicts are expected to
 *                  be frequent and transactions are short.
 *
 *   OPTIMISTIC   - Read the row, check it, UPDATE ... WHERE version=N. Fast
 *                  happy path, no DB locks during reads. Losers get an
 *                  OptimisticLockingFailureException and must retry.
 *                  Best when conflicts are rare.
 *
 *   DISTRIBUTED  - Redis (Redisson) MultiLock across the seat ids, then a plain
 *                  CAS UPDATE. Works across multiple JVMs; also good when you
 *                  want the seat held during an external call (payment) without
 *                  holding an open DB transaction the whole time.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BookingService {

    private final SeatRepository       seatRepo;
    private final BookingRepository    bookingRepo;
    private final DistributedLockService distributedLock;
    private final BookingEventPublisher publisher;

    // Self-reference so that internal calls to @Transactional methods go through
    // the Spring AOP proxy (and therefore participate in transactions).
    // Plain `this.doBookInsideLock(...)` would skip the proxy and run without a tx.
    @Autowired @Lazy
    private BookingService self;

    @Value("${booking.seat-lock-ttl-seconds:60}")
    private long seatLockTtlSeconds;

    // ---------------------------------------------------------------------
    // Strategy 1: Pessimistic DB lock (SELECT ... FOR UPDATE)
    // ---------------------------------------------------------------------
    @Transactional(isolation = Isolation.READ_COMMITTED)
    public BookingResponse bookPessimistic(BookingRequest req, String idempotencyKey) {
        Optional<Booking> existing = findIdempotent(idempotencyKey);
        if (existing.isPresent()) return toResponse(existing.get(), "idempotent replay");

        // Acquires row locks in ID order -> no deadlocks between concurrent bookers.
        List<Seat> seats = seatRepo.findAllByIdForUpdate(sorted(req.getSeatIds()));
        assertAvailable(seats, req.getSeatIds());

        seats.forEach(s -> s.setStatus(SeatStatus.BOOKED));
        seatRepo.saveAll(seats);

        Booking booking = persistBooking(req, seats, idempotencyKey, BookingStatus.CONFIRMED);
        registerAfterCommitEvent(booking, seats);
        return toResponse(booking, "booked via pessimistic lock");
    }

    // ---------------------------------------------------------------------
    // Strategy 2: Optimistic locking (@Version + CAS update)
    // ---------------------------------------------------------------------
    @Transactional
    public BookingResponse bookOptimistic(BookingRequest req, String idempotencyKey) {
        Optional<Booking> existing = findIdempotent(idempotencyKey);
        if (existing.isPresent()) return toResponse(existing.get(), "idempotent replay");

        // The conditional UPDATE is an atomic CAS: it only flips rows that are still
        // AVAILABLE. If the number of rows changed < requested, somebody else won.
        int changed = seatRepo.atomicUpdateStatusIfCurrent(
                sorted(req.getSeatIds()), SeatStatus.AVAILABLE, SeatStatus.BOOKED);
        if (changed != req.getSeatIds().size()) {
            throw new SeatUnavailableException(
                    "Only " + changed + " of " + req.getSeatIds().size() + " seats still available");
        }

        List<Seat> seats = seatRepo.findAllById(req.getSeatIds());
        Booking booking = persistBooking(req, seats, idempotencyKey, BookingStatus.CONFIRMED);
        registerAfterCommitEvent(booking, seats);
        return toResponse(booking, "booked via optimistic CAS");
    }

    // ---------------------------------------------------------------------
    // Strategy 3: Redis distributed lock (cross-JVM)
    // ---------------------------------------------------------------------
    public BookingResponse bookWithDistributedLock(BookingRequest req, String idempotencyKey) {
        // Lock OUTSIDE the transaction so we hold it across the whole booking flow
        // including the payment call (which is async but awaited).
        return distributedLock.withSeatLocks(
                req.getSeatIds(),
                2_000,                         // wait up to 2s to acquire
                seatLockTtlSeconds * 1_000,    // auto-release after TTL as a safety net
                () -> self.doBookInsideLock(req, idempotencyKey));
    }

    // Runs inside the distributed lock. Requires a brand-new transaction so that
    // the transaction boundary is strictly inside the lock boundary.
    // Called via `self` so the proxy (and therefore @Transactional) applies.
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public BookingResponse doBookInsideLock(BookingRequest req, String idempotencyKey) {
        Optional<Booking> existing = findIdempotent(idempotencyKey);
        if (existing.isPresent()) return toResponse(existing.get(), "idempotent replay");

        List<Seat> seats = seatRepo.findAllById(req.getSeatIds());
        assertAvailable(seats, req.getSeatIds());

        seats.forEach(s -> s.setStatus(SeatStatus.BOOKED));
        seatRepo.saveAll(seats);

        Booking booking = persistBooking(req, seats, idempotencyKey, BookingStatus.CONFIRMED);
        registerAfterCommitEvent(booking, seats);
        return toResponse(booking, "booked via redis distributed lock");
    }

    // ---------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------

    private Optional<Booking> findIdempotent(String key) {
        if (key == null || key.isBlank()) return Optional.empty();
        return bookingRepo.findByIdempotencyKey(key);
    }

    private static List<Long> sorted(List<Long> ids) {
        return ids.stream().sorted().collect(Collectors.toList());
    }

    private void assertAvailable(List<Seat> seats, List<Long> requested) {
        if (seats.size() != requested.size()) {
            throw new SeatUnavailableException("Some requested seats do not exist");
        }
        List<String> taken = seats.stream()
                .filter(s -> s.getStatus() != SeatStatus.AVAILABLE)
                .map(Seat::getSeatLabel).toList();
        if (!taken.isEmpty()) {
            throw new SeatUnavailableException("Seats already taken: " + taken);
        }
    }

    private Booking persistBooking(BookingRequest req, List<Seat> seats,
                                   String idempotencyKey, BookingStatus status) {
        Booking b = Booking.builder()
                .userId(req.getUserId())
                .showId(req.getShowId())
                .seatIds(seats.stream().map(s -> s.getId().toString()).collect(Collectors.joining(",")))
                .seatLabels(seats.stream().map(Seat::getSeatLabel).collect(Collectors.joining(",")))
                .status(status)
                .idempotencyKey(idempotencyKey)
                .createdAt(LocalDateTime.now())
                .build();
        return bookingRepo.save(b);
    }

    /**
     * Fires the Kafka event after the DB transaction commits. If the transaction
     * rolls back, the event is never sent - no phantom "confirmed" messages.
     */
    private void registerAfterCommitEvent(Booking booking, List<Seat> seats) {
        BookingEvent event = new BookingEvent(
                booking.getId(), booking.getUserId(), booking.getShowId(),
                seats.stream().map(Seat::getSeatLabel).toList(),
                booking.getStatus(), Instant.now());

        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override public void afterCommit() { publisher.publishConfirmed(event); }
            });
        } else {
            // No transaction (shouldn't happen on these paths, but guard anyway).
            publisher.publishConfirmed(event);
        }
    }

    private BookingResponse toResponse(Booking b, String message) {
        List<String> labels = b.getSeatLabels() == null
                ? List.of() : List.of(b.getSeatLabels().split(","));
        return BookingResponse.builder()
                .bookingId(b.getId())
                .showId(b.getShowId())
                .seatLabels(labels)
                .status(b.getStatus())
                .message(message)
                .build();
    }
}
