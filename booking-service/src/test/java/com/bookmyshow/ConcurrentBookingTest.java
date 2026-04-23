package com.bookmyshow;

import com.bookmyshow.dto.BookingRequest;
import com.bookmyshow.service.BookingService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Proof that exactly ONE booking wins when 100 threads fight for the same seat.
 *
 * Uses H2 + embedded Kafka + an embedded Redis would require more setup, so this
 * test runs with the profile where Redis/Kafka/distributed-lock are not exercised:
 * only the pure-DB strategies (optimistic, pessimistic) are verified here.
 *
 * Run the distributed-lock variant via the /api/loadtest endpoint with Redis up.
 */
@SpringBootTest
@TestPropertySource(properties = {
        "spring.kafka.bootstrap-servers=localhost:9092",
        "spring.autoconfigure.exclude=" +
                "org.springframework.boot.autoconfigure.kafka.KafkaAutoConfiguration," +
                "org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration," +
                "org.redisson.spring.starter.RedissonAutoConfiguration," +
                "org.springframework.boot.autoconfigure.cache.CacheAutoConfiguration"
})
class ConcurrentBookingTest {

    @Autowired BookingService bookings;

    @Test
    void optimisticLocking_exactlyOneWinner() throws Exception {
        runRace("optimistic", 1L);
    }

    @Test
    void pessimisticLocking_exactlyOneWinner() throws Exception {
        runRace("pessimistic", 2L);
    }

    private void runRace(String strategy, long seatId) throws Exception {
        int threads = 100;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        AtomicInteger wins = new AtomicInteger();

        for (int i = 0; i < threads; i++) {
            final int u = i + 1;
            pool.submit(() -> {
                try {
                    start.await();
                    BookingRequest req = new BookingRequest();
                    req.setUserId((long) u);
                    req.setShowId(1L);
                    req.setSeatIds(List.of(seatId));
                    String idem = UUID.randomUUID().toString();
                    if ("pessimistic".equals(strategy)) bookings.bookPessimistic(req, idem);
                    else                                 bookings.bookOptimistic(req, idem);
                    wins.incrementAndGet();
                } catch (Exception ignored) {
                    // losers throw - that's fine
                } finally { done.countDown(); }
            });
        }
        start.countDown();
        assertThat(done.await(30, TimeUnit.SECONDS)).isTrue();
        pool.shutdownNow();

        assertThat(wins.get())
                .as("Exactly one booking must succeed for strategy=%s", strategy)
                .isEqualTo(1);
    }
}
