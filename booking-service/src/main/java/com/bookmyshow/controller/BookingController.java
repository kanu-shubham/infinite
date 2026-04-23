package com.bookmyshow.controller;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.dto.BookingResponse;
import com.bookmyshow.exception.RateLimitExceededException;
import com.bookmyshow.ratelimit.RedisRateLimiter;
import com.bookmyshow.service.BookingService;
import com.bookmyshow.service.IdempotencyService;
import com.bookmyshow.service.PaymentService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

@RestController
@RequestMapping("/api/bookings")
@RequiredArgsConstructor
public class BookingController {

    private final BookingService      bookings;
    private final PaymentService      payments;
    private final IdempotencyService  idempotency;
    private final RedisRateLimiter    rateLimiter;

    /**
     * Sync booking. Demonstrates all three concurrency strategies - pass
     * ?strategy=optimistic|pessimistic|distributed (default: optimistic).
     *
     * Rate limit: 20 booking attempts / minute per user.
     */
    @PostMapping
    public ResponseEntity<BookingResponse> book(
            @Valid @RequestBody BookingRequest req,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey,
            @RequestParam(value = "strategy", required = false) String strategyOverride) {

        if (!rateLimiter.allow("user:" + req.getUserId(), 20, 60_000)) {
            throw new RateLimitExceededException("Too many booking attempts, slow down");
        }
        if (!idempotency.tryClaim(idempotencyKey)) {
            // Key already used - BookingService will return the stored result.
        }

        String strategy = strategyOverride != null ? strategyOverride
                : (req.getStrategy() != null ? req.getStrategy() : "optimistic");

        BookingResponse resp = switch (strategy.toLowerCase()) {
            case "pessimistic" -> bookings.bookPessimistic(req, idempotencyKey);
            case "distributed" -> bookings.bookWithDistributedLock(req, idempotencyKey);
            default             -> bookings.bookOptimistic(req, idempotencyKey);
        };
        return ResponseEntity.ok(resp);
    }

    /**
     * Async booking: HTTP thread returns immediately, payment runs on the payment pool.
     *
     * Returning a CompletableFuture from a @RestController method makes Spring MVC do
     * async dispatch - the Tomcat thread is released and the response is sent once the
     * future completes. This is how you keep 200 Tomcat threads serving 10,000 pending
     * bookings.
     */
    @PostMapping("/async")
    public CompletableFuture<ResponseEntity<BookingResponse>> bookAsync(
            @Valid @RequestBody BookingRequest req,
            @RequestHeader(value = "Idempotency-Key", required = false) String idempotencyKey) {

        if (!rateLimiter.allow("user:" + req.getUserId(), 20, 60_000)) {
            throw new RateLimitExceededException("Too many booking attempts, slow down");
        }
        idempotency.tryClaim(idempotencyKey);

        // 1. Reserve seats (fast DB path, optimistic CAS).
        BookingResponse reserved = bookings.bookOptimistic(req, idempotencyKey);

        // 2. Charge the card asynchronously. Chain the result back into an HTTP response.
        return payments.charge(req.getUserId(), 500L * req.getSeatIds().size())
                .thenApply(result -> {
                    if (result.success()) {
                        reserved.setMessage("Confirmed. Payment ref: " + result.reference());
                        return ResponseEntity.ok(reserved);
                    }
                    reserved.setMessage("Payment failed: " + result.reference());
                    return ResponseEntity.status(402).body(reserved);
                })
                .exceptionally(ex -> {
                    Throwable cause = (ex instanceof CompletionException && ex.getCause() != null)
                            ? ex.getCause() : ex;
                    reserved.setMessage("Error: " + cause.getMessage());
                    return ResponseEntity.status(500).body(reserved);
                });
    }
}
