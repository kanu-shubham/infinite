package com.bookmyshow.readmodel;

import com.bookmyshow.events.PaymentChargedEvent;
import com.bookmyshow.events.PaymentFailedEvent;
import com.bookmyshow.events.SagaTopics;
import com.bookmyshow.events.SeatsReservedEvent;
import com.bookmyshow.repository.SeatRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;

/**
 * Keeps the read model in sync by listening to saga events.
 *
 * The read-model table is eventually consistent - there is a lag (usually < 100ms) between
 * the write model changing and this projector updating. For our UI ("X seats left"),
 * that's fine. For the authoritative "can I book this seat?" check, we still use the
 * write model (the Seat table with locks).
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ShowAvailabilityProjector {

    private final ShowAvailabilityRepository readRepo;
    private final SeatRepository seatRepo;

    @KafkaListener(
            topics = {SagaTopics.SEATS_RESERVED, SagaTopics.PAYMENT_CHARGED, SagaTopics.PAYMENT_FAILED},
            groupId = "show-availability-projector",
            concurrency = "1"  // projector is single-threaded; concurrency would require partition-aware dedupe
    )
    @Transactional
    public void onSagaEvent(Object event) {
        Long showId = null;
        if (event instanceof SeatsReservedEvent e)       showId = e.getShowId();
        else if (event instanceof PaymentChargedEvent)   showId = null; // no showId; recompute from booking
        else if (event instanceof PaymentFailedEvent)    showId = null;
        if (showId == null) return;  // keep it simple for the demo

        recomputeFor(showId);
    }

    /**
     * Recompute from the source of truth. In production you'd incrementally update
     * (`availableCount -= seatsHeld`), but a full recompute is simpler and avoids drift.
     * Even for a 200-seat show this is just a few hundred microseconds.
     */
    public void recomputeFor(Long showId) {
        var seats = seatRepo.findByShowId(showId);
        int total = seats.size();
        int avail = 0, held = 0, booked = 0;
        for (var s : seats) {
            switch (s.getStatus()) {
                case AVAILABLE -> avail++;
                case HELD      -> held++;
                case BOOKED    -> booked++;
            }
        }
        ShowAvailability row = readRepo.findById(showId)
                .orElse(ShowAvailability.builder().showId(showId).build());
        row.setAvailableCount(avail);
        row.setHeldCount(held);
        row.setBookedCount(booked);
        row.setTotalCount(total);
        row.setUpdatedAt(Instant.now());
        readRepo.save(row);

        log.debug("Projected showId={} avail={} held={} booked={}", showId, avail, held, booked);
    }
}
