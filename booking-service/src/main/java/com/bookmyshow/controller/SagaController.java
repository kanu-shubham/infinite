package com.bookmyshow.controller;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.entity.Booking;
import com.bookmyshow.readmodel.ShowAvailability;
import com.bookmyshow.readmodel.ShowAvailabilityRepository;
import com.bookmyshow.repository.BookingRepository;
import com.bookmyshow.saga.BookingSagaCoordinator;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * REST entry point for the choreography saga.
 *
 *   POST /api/saga/bookings    -> kicks off the saga, returns 202 with bookingId.
 *                                 Booking is PENDING; payment runs asynchronously
 *                                 across services. Poll the GET endpoint to see status.
 *
 *   GET /api/saga/bookings/{id}
 *
 *   GET /api/shows/{id}/availability  -> served from the read model (CQRS demo).
 */
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class SagaController {

    private final BookingSagaCoordinator saga;
    private final BookingRepository bookings;
    private final ShowAvailabilityRepository readModel;

    @PostMapping("/saga/bookings")
    public ResponseEntity<Map<String, Object>> start(
            @Valid @RequestBody BookingRequest req,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey) {

        long amountCents = 500L * req.getSeatIds().size();
        Long id = saga.startSaga(req.getUserId(), req.getShowId(), req.getSeatIds(),
                                  amountCents, idempotencyKey);
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(Map.of(
                "bookingId", id,
                "status", "PENDING",
                "message", "Saga started; payment in progress. Poll for status."));
    }

    @GetMapping("/saga/bookings/{id}")
    public ResponseEntity<?> get(@PathVariable Long id) {
        return bookings.findById(id)
                .<ResponseEntity<?>>map(b -> ResponseEntity.ok(Map.of(
                        "bookingId", b.getId(),
                        "status", b.getStatus(),
                        "seatLabels", b.getSeatLabels())))
                .orElse(ResponseEntity.notFound().build());
    }

    /** Read-model query: served from the denormalized show_availability table. */
    @GetMapping("/shows/{id}/availability")
    public ResponseEntity<ShowAvailability> availability(@PathVariable Long id) {
        return readModel.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
}
