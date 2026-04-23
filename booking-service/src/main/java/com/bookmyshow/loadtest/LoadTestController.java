package com.bookmyshow.loadtest;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.dto.BookingResponse;
import com.bookmyshow.service.BookingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Built-in load test endpoint.
 *
 * POST /api/loadtest?strategy=pessimistic&threads=100&seatId=1
 *
 * Launches N threads that all try to book the SAME seat simultaneously, using a
 * CountDownLatch to release them at the exact same moment (real "thundering herd").
 * Exactly ONE thread must succeed - if the number is different for a given strategy,
 * that strategy has a concurrency bug.
 *
 * This is a great teaching tool: run it against each strategy and watch the output.
 */
@Slf4j
@RestController
@RequestMapping("/api/loadtest")
@RequiredArgsConstructor
public class LoadTestController {

    private final BookingService bookings;

    @PostMapping
    public Map<String, Object> hammer(
            @RequestParam(defaultValue = "optimistic") String strategy,
            @RequestParam(defaultValue = "100") int threads,
            @RequestParam(defaultValue = "1") long seatId) throws InterruptedException {

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);  // gate so all threads start together
        CountDownLatch done  = new CountDownLatch(threads);

        AtomicInteger success = new AtomicInteger();
        AtomicInteger conflict = new AtomicInteger();
        AtomicInteger errors = new AtomicInteger();

        long t0 = System.currentTimeMillis();

        for (int i = 0; i < threads; i++) {
            final int userId = i + 1;
            pool.submit(() -> {
                try {
                    start.await();
                    BookingRequest req = new BookingRequest();
                    req.setUserId((long) userId);
                    req.setShowId(1L);
                    req.setSeatIds(List.of(seatId));
                    String idem = UUID.randomUUID().toString();

                    BookingResponse r = switch (strategy) {
                        case "pessimistic" -> bookings.bookPessimistic(req, idem);
                        case "distributed" -> bookings.bookWithDistributedLock(req, idem);
                        default             -> bookings.bookOptimistic(req, idem);
                    };
                    if (r != null) success.incrementAndGet();
                } catch (com.bookmyshow.exception.SeatUnavailableException
                         | org.springframework.dao.OptimisticLockingFailureException e) {
                    conflict.incrementAndGet();
                } catch (Exception e) {
                    errors.incrementAndGet();
                    log.debug("Load-test unexpected error", e);
                } finally {
                    done.countDown();
                }
            });
        }

        start.countDown();          // GO!
        done.await(30, TimeUnit.SECONDS);
        pool.shutdownNow();

        long elapsed = System.currentTimeMillis() - t0;
        return Map.of(
                "strategy", strategy,
                "threads",  threads,
                "seatId",   seatId,
                "success",  success.get(),   // MUST be 1 for a correct strategy
                "conflict", conflict.get(),  // expected: threads - 1
                "errors",   errors.get(),
                "elapsedMs", elapsed);
    }
}
