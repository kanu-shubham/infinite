package com.bookmyshow.broken;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.dto.BookingResponse;
import com.bookmyshow.entity.Booking;
import com.bookmyshow.entity.BookingStatus;
import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.SeatStatus;
import com.bookmyshow.exception.SeatUnavailableException;
import com.bookmyshow.repository.BookingRepository;
import com.bookmyshow.repository.SeatRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

/**
 * INTENTIONALLY BROKEN booking service used for teaching.
 *
 * It uses Java's `synchronized` keyword as the only concurrency control,
 * with NO database-level locking and NO @Version optimistic CAS.
 *
 * Inside ONE JVM:
 *   - synchronized works. The 100-thread load test against ONE instance returns success=1.
 *
 * Across MULTIPLE JVMs (e.g. `docker compose up --scale booking-service=2`):
 *   - synchronized only locks within its own JVM.
 *   - JVM-A and JVM-B both grab their own monitor, both check the seat, both write.
 *   - You get success > 1. Your show is oversold. The bug is real and reproducible.
 *
 * Run /api/loadtest?strategy=broken-synchronized against a single instance vs. two
 * instances behind nginx to see this empirically.
 */
@Service
@RequiredArgsConstructor
public class SynchronizedBookingService {

    private final SeatRepository seatRepo;
    private final BookingRepository bookingRepo;

    // Single shared lock object. All threads in this JVM serialize on it.
    // Crucially, JVM-B has its OWN copy of this field — it cannot see JVM-A's lock.
    private final Object lock = new Object();

    @Transactional
    public BookingResponse bookBroken(BookingRequest req, String idempotencyKey) {
        synchronized (lock) {
            List<Long> ids = req.getSeatIds().stream().sorted().collect(Collectors.toList());
            List<Seat> seats = seatRepo.findAllById(ids);

            if (seats.size() != ids.size()) {
                throw new SeatUnavailableException("Some seats do not exist");
            }
            for (Seat s : seats) {
                if (s.getStatus() != SeatStatus.AVAILABLE) {
                    throw new SeatUnavailableException("Seat " + s.getSeatLabel() + " not available");
                }
            }
            // PROBLEM: between the check above and the save below, another JVM can also pass
            // its check (synchronized doesn't reach across JVMs) and we both write BOOKED.
            seats.forEach(s -> s.setStatus(SeatStatus.BOOKED));
            seatRepo.saveAll(seats);

            Booking b = Booking.builder()
                    .userId(req.getUserId())
                    .showId(req.getShowId())
                    .seatIds(seats.stream().map(s -> s.getId().toString()).collect(Collectors.joining(",")))
                    .seatLabels(seats.stream().map(Seat::getSeatLabel).collect(Collectors.joining(",")))
                    .status(BookingStatus.CONFIRMED)
                    .idempotencyKey(idempotencyKey)
                    .createdAt(LocalDateTime.now())
                    .build();
            bookingRepo.save(b);

            return BookingResponse.builder()
                    .bookingId(b.getId())
                    .showId(b.getShowId())
                    .seatLabels(seats.stream().map(Seat::getSeatLabel).toList())
                    .status(BookingStatus.CONFIRMED)
                    .message("booked via INTENTIONALLY BROKEN synchronized")
                    .build();
        }
    }
}
