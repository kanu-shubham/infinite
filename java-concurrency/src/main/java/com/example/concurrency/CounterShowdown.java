package com.example.concurrency;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pattern 3: Race three counter implementations under heavy contention.
 *
 *   1. Plain int  - fastest but loses updates because ++ is not atomic.
 *   2. synchronized - correct, but threads serialize on the monitor.
 *   3. AtomicInteger - correct AND lock-free; uses a CPU CAS instruction.
 *
 * Expected output: the unsynchronized counter ends well below
 * THREADS * INCREMENTS_PER_THREAD (proving the race), while the other two
 * land exactly on it. AtomicInteger is typically meaningfully faster than
 * synchronized under high contention.
 */
public final class CounterShowdown {

    private static final int THREADS = 8;
    private static final int INCREMENTS_PER_THREAD = 1_000_000;
    private static final long EXPECTED = (long) THREADS * INCREMENTS_PER_THREAD;

    public static void main(String[] args) throws InterruptedException {
        // JVM warm-up so the JIT has compiled hot loops before we measure.
        run("warmup-unsync",  new UnsynchronizedCounter());
        run("warmup-sync",    new SynchronizedCounter());
        run("warmup-atomic",  new AtomicCounter());

        System.out.printf("%nThreads=%d, increments/thread=%,d, expected total=%,d%n%n",
            THREADS, INCREMENTS_PER_THREAD, EXPECTED);
        System.out.printf("%-20s %12s %12s %12s%n", "counter", "final", "lost", "elapsed-ms");
        System.out.println("-".repeat(60));

        run("Unsynchronized", new UnsynchronizedCounter());
        run("Synchronized",   new SynchronizedCounter());
        run("AtomicInteger",  new AtomicCounter());
    }

    private static void run(String label, Counter counter) throws InterruptedException {
        ExecutorService pool = Executors.newFixedThreadPool(THREADS);
        long t0 = System.nanoTime();
        for (int i = 0; i < THREADS; i++) {
            pool.execute(() -> {
                for (int j = 0; j < INCREMENTS_PER_THREAD; j++) {
                    counter.increment();
                }
            });
        }
        pool.shutdown();
        if (!pool.awaitTermination(60, TimeUnit.SECONDS)) {
            pool.shutdownNow();
        }
        long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

        if (label.startsWith("warmup")) return;

        long got = counter.get();
        long lost = EXPECTED - got;
        System.out.printf("%-20s %12d %12d %12d%n", label, got, lost, elapsedMs);
    }

    private interface Counter {
        void increment();
        long get();
    }

    private static final class UnsynchronizedCounter implements Counter {
        private int value;
        public void increment() { value++; }      // read-modify-write race
        public long get()       { return value; }
    }

    private static final class SynchronizedCounter implements Counter {
        private int value;
        public synchronized void increment() { value++; }
        public synchronized long get()       { return value; }
    }

    private static final class AtomicCounter implements Counter {
        private final AtomicInteger value = new AtomicInteger();
        public void increment() { value.incrementAndGet(); }
        public long get()       { return value.get(); }
    }
}
