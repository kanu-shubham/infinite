package com.bookmyshow.service;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Simulates a payment gateway call.
 *
 * Two things to notice:
 *   1. @Async("paymentExecutor") - this method runs on the bounded pool defined in AsyncConfig,
 *      NOT on the Tomcat request thread. The HTTP thread returns immediately and can handle the
 *      next request. This is how you keep a 200-thread server from being blocked by 200 slow calls.
 *   2. The return type is CompletableFuture so callers can chain .thenApply / .thenCompose.
 */
@Slf4j
@Service
public class PaymentService {

    @Value("${booking.payment.simulated-latency-ms:200}")
    private long simulatedLatencyMs;

    @Async("paymentExecutor")
    public CompletableFuture<PaymentResult> charge(Long userId, long amountCents) {
        try {
            Thread.sleep(simulatedLatencyMs);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return CompletableFuture.completedFuture(new PaymentResult(false, "interrupted"));
        }
        // 95% success in simulation so you can see both paths in load tests.
        boolean ok = ThreadLocalRandom.current().nextInt(100) < 95;
        String txnId = ok ? "txn_" + System.nanoTime() : null;
        log.debug("Payment for user={} amount={} ok={} txn={}", userId, amountCents, ok, txnId);
        return CompletableFuture.completedFuture(
                new PaymentResult(ok, ok ? txnId : "declined"));
    }

    public record PaymentResult(boolean success, String reference) {}
}
