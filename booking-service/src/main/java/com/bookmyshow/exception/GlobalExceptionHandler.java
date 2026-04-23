package com.bookmyshow.exception;

import org.springframework.dao.OptimisticLockingFailureException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.Map;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(SeatUnavailableException.class)
    public ResponseEntity<Map<String, Object>> handleSeatTaken(SeatUnavailableException e) {
        return ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of(
                "error", "SEAT_UNAVAILABLE",
                "message", e.getMessage()));
    }

    @ExceptionHandler(OptimisticLockingFailureException.class)
    public ResponseEntity<Map<String, Object>> handleOptimistic(OptimisticLockingFailureException e) {
        // Surfaced when two transactions both read version=N and both try to update.
        return ResponseEntity.status(HttpStatus.CONFLICT).body(Map.of(
                "error", "CONCURRENT_MODIFICATION",
                "message", "Another booker grabbed the seat first, please retry."));
    }

    @ExceptionHandler(RateLimitExceededException.class)
    public ResponseEntity<Map<String, Object>> handleRate(RateLimitExceededException e) {
        return ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS).body(Map.of(
                "error", "RATE_LIMITED",
                "message", e.getMessage()));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> handleBadReq(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(Map.of(
                "error", "BAD_REQUEST",
                "message", e.getMessage()));
    }
}
