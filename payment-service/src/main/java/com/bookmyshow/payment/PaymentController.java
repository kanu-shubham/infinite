package com.bookmyshow.payment;

import com.bookmyshow.payment.entity.Payment;
import com.bookmyshow.payment.repository.PaymentRepository;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/** Inspection endpoints — useful when you're poking a saga from the command line. */
@RestController
@RequestMapping("/payments")
@RequiredArgsConstructor
public class PaymentController {

    private final PaymentRepository repo;
    private final CircuitBreakerRegistry breakers;

    @GetMapping("/booking/{bookingId}")
    public Payment forBooking(@PathVariable Long bookingId) {
        return repo.findByBookingId(bookingId).orElse(null);
    }

    /** Live circuit breaker status — also visible via /actuator/circuitbreakers. */
    @GetMapping("/breaker")
    public Map<String, Object> breakerState() {
        var cb = breakers.circuitBreaker("paymentGateway");
        var m  = cb.getMetrics();
        return Map.of(
                "state", cb.getState().toString(),
                "failureRate", m.getFailureRate(),
                "slowCallRate", m.getSlowCallRate(),
                "bufferedCalls", m.getNumberOfBufferedCalls(),
                "failedCalls", m.getNumberOfFailedCalls(),
                "successfulCalls", m.getNumberOfSuccessfulCalls());
    }
}
