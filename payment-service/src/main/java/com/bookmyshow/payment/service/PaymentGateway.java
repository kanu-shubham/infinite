package com.bookmyshow.payment.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Simulated external payment gateway.
 *
 * Two production-grade behaviors are illustrated here:
 *
 *   @CircuitBreaker(name = "paymentGateway", fallbackMethod = "...")
 *      - Resilience4j tracks success/failure of calls to this method.
 *      - After 50% failure over a sliding window of 10 calls, breaker OPENS:
 *        every subsequent call short-circuits to the fallback for 5 seconds.
 *      - It then HALF-OPENS, allows 3 probe calls, and closes again if they succeed.
 *      - Result: when the gateway is broken, your service fails FAST instead of piling
 *        up threads on a 30-second timeout cascade.
 *
 *   @Retry(name = "paymentGateway")
 *      - Transient errors are retried with backoff. Retry runs INSIDE the breaker so
 *        the breaker only sees the final outcome (avoids inflating failure counts).
 *
 * Toggle the failure mode at runtime via the env var PAYMENT_FAIL_MODE:
 *   none   - happy path
 *   always - throws every time (drives the breaker open)
 *   slow   - sleeps 5s before answering (combined with TimeLimiter, would time out)
 *   random - 30% failure rate
 */
@Slf4j
@Component
public class PaymentGateway {

    @Value("${payment.fail-mode:none}")
    private String failMode;

    @Value("${payment.slow-delay-ms:5000}")
    private long slowDelayMs;

    @CircuitBreaker(name = "paymentGateway", fallbackMethod = "fallback")
    @Retry(name = "paymentGateway")
    public String charge(Long userId, long amountCents) {
        switch (failMode) {
            case "always" -> throw new PaymentGatewayException("PAYMENT_FAIL_MODE=always");
            case "slow"   -> sleep(slowDelayMs);
            case "random" -> {
                if (ThreadLocalRandom.current().nextInt(100) < 30) {
                    throw new PaymentGatewayException("simulated transient error");
                }
                sleep(150);   // typical happy-path latency
            }
            default -> sleep(80);
        }
        return "txn_" + System.nanoTime();
    }

    /**
     * Fallback signature: same args as the protected method PLUS the Throwable.
     * Resilience4j picks the most specific match.
     */
    @SuppressWarnings("unused")
    private String fallback(Long userId, long amountCents, Throwable t) {
        log.warn("CircuitBreaker fallback for user={} amount={}: {}",
                 userId, amountCents, t.getClass().getSimpleName());
        // Re-throw to propagate as a saga-level failure. Some teams return a sentinel here
        // and decide downstream; this project keeps it explicit.
        throw new PaymentGatewayException("gateway unavailable: " + t.getMessage());
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
