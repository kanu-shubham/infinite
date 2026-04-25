package com.example.concurrency;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pattern 4: Fire N HTTP requests at a public API "simultaneously" and
 * measure latency. The trick is the START gate.
 *
 * If we just submit 50 tasks to a thread pool, the first ones start running
 * before the last ones are even queued. To get a true thundering-herd,
 * every worker:
 *   1. Is created and parked on `startGate.await()`.
 *   2. Records its start time only AFTER the gate opens.
 *   3. Counts down `doneGate` when finished.
 *
 * The main thread opens the gate with a single countDown(), then waits on
 * `doneGate.await()` for everyone to finish.
 */
public final class ApiBenchmark {

    private static final String TARGET_URL = "https://httpbin.org/get";
    private static final int REQUEST_COUNT = 50;
    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(15);

    public static void main(String[] args) throws InterruptedException {
        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .version(HttpClient.Version.HTTP_2)
            .build();

        ExecutorService pool = Executors.newFixedThreadPool(REQUEST_COUNT);
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch doneGate  = new CountDownLatch(REQUEST_COUNT);

        long[] latencies = new long[REQUEST_COUNT];
        Arrays.fill(latencies, -1);
        AtomicInteger successes = new AtomicInteger();
        AtomicInteger failures  = new AtomicInteger();

        for (int i = 0; i < REQUEST_COUNT; i++) {
            final int idx = i;
            pool.execute(() -> {
                try {
                    startGate.await();
                    long t0 = System.nanoTime();
                    HttpRequest req = HttpRequest.newBuilder(URI.create(TARGET_URL))
                        .timeout(REQUEST_TIMEOUT)
                        .GET()
                        .build();
                    HttpResponse<Void> resp = client.send(req, HttpResponse.BodyHandlers.discarding());
                    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
                    latencies[idx] = elapsedMs;
                    if (resp.statusCode() / 100 == 2) {
                        successes.incrementAndGet();
                    } else {
                        failures.incrementAndGet();
                    }
                } catch (Exception e) {
                    failures.incrementAndGet();
                } finally {
                    doneGate.countDown();
                }
            });
        }

        System.out.printf("Firing %d requests at %s ...%n", REQUEST_COUNT, TARGET_URL);
        long wallStart = System.nanoTime();
        startGate.countDown();          // open the floodgate
        doneGate.await();               // wait for everyone to finish
        long wallMs = (System.nanoTime() - wallStart) / 1_000_000;

        pool.shutdown();
        pool.awaitTermination(5, TimeUnit.SECONDS);

        printStats(latencies, successes.get(), failures.get(), wallMs);
    }

    private static void printStats(long[] latencies, int ok, int fail, long wallMs) {
        long[] sorted = Arrays.stream(latencies).filter(l -> l >= 0).sorted().toArray();
        if (sorted.length == 0) {
            System.out.println("No successful timings recorded.");
            return;
        }
        long min = sorted[0];
        long max = sorted[sorted.length - 1];
        double avg = Arrays.stream(sorted).average().orElse(0);
        long p50 = sorted[(int) (sorted.length * 0.50)];
        long p95 = sorted[(int) Math.min(sorted.length - 1, sorted.length * 0.95)];
        long p99 = sorted[(int) Math.min(sorted.length - 1, sorted.length * 0.99)];

        System.out.println();
        System.out.printf("Wall-clock time : %d ms%n", wallMs);
        System.out.printf("Successes/Fails : %d / %d%n", ok, fail);
        System.out.printf("Latency min/avg : %d / %.1f ms%n", min, avg);
        System.out.printf("Latency p50/p95 : %d / %d ms%n", p50, p95);
        System.out.printf("Latency p99/max : %d / %d ms%n", p99, max);
    }
}
