package com.example.concurrency;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Pattern 1: Parallel file downloader using a fixed-size ExecutorService.
 *
 * Why a thread pool? File downloads are I/O-bound: each thread spends most of
 * its time waiting on the network, not on the CPU. A fixed pool with N workers
 * lets us keep N requests in flight simultaneously without spawning a thread
 * per URL, which would be wasteful and unbounded.
 *
 * Key concepts demonstrated:
 *   - Executors.newFixedThreadPool(n)
 *   - Submitting Callables and collecting Futures
 *   - Graceful shutdown with awaitTermination
 */
public final class ParallelFileDownloader {

    private static final List<String> URLS = List.of(
        "https://httpbin.org/bytes/10000",
        "https://httpbin.org/bytes/20000",
        "https://httpbin.org/bytes/30000",
        "https://httpbin.org/bytes/40000",
        "https://httpbin.org/bytes/50000",
        "https://httpbin.org/bytes/60000",
        "https://httpbin.org/bytes/70000",
        "https://httpbin.org/bytes/80000",
        "https://httpbin.org/bytes/90000",
        "https://httpbin.org/bytes/100000"
    );

    private static final Path DOWNLOAD_DIR = Paths.get("downloads");
    private static final int POOL_SIZE = 10;

    public static void main(String[] args) throws Exception {
        Files.createDirectories(DOWNLOAD_DIR);

        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();

        ExecutorService pool = Executors.newFixedThreadPool(POOL_SIZE);

        long start = System.currentTimeMillis();
        try {
            List<Future<DownloadResult>> futures = pool.invokeAll(buildTasks(client));
            for (Future<DownloadResult> f : futures) {
                try {
                    DownloadResult r = f.get();
                    System.out.printf("OK   %-50s %6d bytes  %4d ms%n",
                        r.url, r.bytes, r.elapsedMs);
                } catch (ExecutionException e) {
                    System.out.printf("FAIL %s -> %s%n", "(unknown)", e.getCause());
                }
            }
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(30, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
        }
        long total = System.currentTimeMillis() - start;
        System.out.printf("%nTotal wall-clock time: %d ms (pool size %d)%n", total, POOL_SIZE);
    }

    private static List<Callable<DownloadResult>> buildTasks(HttpClient client) {
        return URLS.stream()
            .<Callable<DownloadResult>>map(url -> () -> download(client, url))
            .toList();
    }

    private static DownloadResult download(HttpClient client, String url) throws IOException, InterruptedException {
        long t0 = System.currentTimeMillis();
        HttpRequest req = HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofSeconds(20))
            .GET()
            .build();
        HttpResponse<InputStream> resp = client.send(req, HttpResponse.BodyHandlers.ofInputStream());

        String fileName = url.replaceAll("[^a-zA-Z0-9.-]", "_");
        Path target = DOWNLOAD_DIR.resolve(fileName);
        long bytes;
        try (InputStream in = resp.body()) {
            bytes = Files.copy(in, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
        return new DownloadResult(url, bytes, System.currentTimeMillis() - t0);
    }

    private record DownloadResult(String url, long bytes, long elapsedMs) {}
}
