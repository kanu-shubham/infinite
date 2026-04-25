# Java Concurrency Patterns

A small, self-contained learning project that demonstrates four foundational
concurrency patterns in modern Java. Each pattern is one file, runnable on
its own.

| # | Pattern | Class | Building blocks |
|---|---------|-------|-----------------|
| 1 | Parallel file downloader | `ParallelFileDownloader` | `ExecutorService`, `Callable`, `Future` |
| 2 | Producer / consumer       | `ProducerConsumer`        | `BlockingQueue`, poison-pill sentinel |
| 3 | Counter showdown          | `CounterShowdown`         | `synchronized`, `AtomicInteger`, race conditions |
| 4 | API benchmark             | `ApiBenchmark`            | `CountDownLatch`, latency percentiles |

Requires JDK 17+ (uses `record` and `java.net.http.HttpClient`).

## Build & run

```bash
cd java-concurrency
mkdir -p out
javac -d out $(find src/main/java -name '*.java')

java -cp out com.example.concurrency.ParallelFileDownloader
java -cp out com.example.concurrency.ProducerConsumer
java -cp out com.example.concurrency.CounterShowdown
java -cp out com.example.concurrency.ApiBenchmark
```

Patterns 1 and 4 require internet access (they hit `httpbin.org`).

## What each example teaches

### 1. Parallel file downloader (`ExecutorService`)

Downloads ten files concurrently from `httpbin.org`. The point is that
because downloads are I/O-bound, ten threads finish in roughly the time of
the *slowest single download* — not the sum.

Key idea: `Executors.newFixedThreadPool(N)` gives you a bounded worker pool.
`invokeAll` submits every task and returns once they all complete. Always
shut the pool down in a `finally` block.

### 2. Producer / consumer (`BlockingQueue`)

One thread produces random integers, another consumes them, and they share a
bounded `ArrayBlockingQueue`. The queue itself handles all synchronization:

- `put()` blocks the producer when the queue is full → automatic back-pressure.
- `take()` blocks the consumer when the queue is empty → no busy-waiting.

A "poison pill" (`Integer.MIN_VALUE`) tells the consumer to stop. This is
much simpler — and harder to get wrong — than rolling your own
`wait()` / `notify()` loop.

### 3. Counter showdown (race conditions, `synchronized`, `AtomicInteger`)

Eight threads each increment a shared counter one million times. Three
implementations race:

| Counter         | Correct? | Why |
|-----------------|----------|-----|
| `int value++`              | **No**  | `++` is read–modify–write, three separate steps; threads overwrite each other and updates are silently lost. |
| `synchronized`             | Yes     | One thread at a time holds the monitor; safe but threads serialize on it under contention. |
| `AtomicInteger`            | Yes     | Uses a CPU-level CAS instruction; lock-free and typically faster than `synchronized` under contention. |

Sample run (your numbers will vary):

```
counter                     final         lost   elapsed-ms
------------------------------------------------------------
Unsynchronized            3142219      4857781           38   ← lost ~60%!
Synchronized              8000000            0          412
AtomicInteger             8000000            0          181
```

The takeaway: `++` on a shared field is **not** atomic. Always use
`synchronized` or an `Atomic*` class.

### 4. API benchmark (`CountDownLatch`)

Fires fifty HTTP GETs at `httpbin.org/get` "simultaneously" and prints
min / avg / p50 / p95 / p99 / max latencies.

The interesting part is the **two-latch start gate**:

```
startGate = new CountDownLatch(1);    // closed; held by main
doneGate  = new CountDownLatch(N);    // counts each finished worker

// each worker:
startGate.await();    // park until main opens the gate
... do request ...
doneGate.countDown();

// main:
startGate.countDown();   // releases all 50 at once
doneGate.await();        // wait for them all to finish
```

Without the start gate the first tasks scheduled would start running before
the last ones were even submitted, smearing the load over time and giving
misleading latency numbers. With it, all 50 workers race out of the gate
together and you get a real "thundering herd" benchmark.

## File layout

```
java-concurrency/
├── README.md
└── src/main/java/com/example/concurrency/
    ├── ParallelFileDownloader.java
    ├── ProducerConsumer.java
    ├── CounterShowdown.java
    └── ApiBenchmark.java
```
