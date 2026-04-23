package com.bookmyshow.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Thread pools.
 *
 * Two pools on purpose:
 *   paymentExecutor   - simulates a slow downstream call (payment gateway). Bounded queue so
 *                       backpressure surfaces as a rejected execution instead of an OOM.
 *   notificationExecutor - fire-and-forget work (email / SMS). Unbounded-ish queue, small pool.
 *
 * Sizing rule of thumb (Little's law): pool_size >= target_throughput_rps * avg_latency_seconds.
 * 20 threads * (1 / 0.2s) = 100 rps of payment capacity.
 */
@Configuration
public class AsyncConfig {

    @Value("${booking.payment.thread-pool-size:20}")
    private int paymentPoolSize;

    @Bean("paymentExecutor")
    public Executor paymentExecutor() {
        ThreadPoolTaskExecutor ex = new ThreadPoolTaskExecutor();
        ex.setCorePoolSize(paymentPoolSize);
        ex.setMaxPoolSize(paymentPoolSize);
        ex.setQueueCapacity(200);
        ex.setThreadNamePrefix("payment-");
        // CallerRunsPolicy: when the queue is full, the calling (Tomcat) thread runs the task
        // itself. This naturally throttles incoming traffic instead of silently dropping work.
        ex.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        ex.initialize();
        return ex;
    }

    @Bean("notificationExecutor")
    public Executor notificationExecutor() {
        ThreadPoolTaskExecutor ex = new ThreadPoolTaskExecutor();
        ex.setCorePoolSize(4);
        ex.setMaxPoolSize(8);
        ex.setQueueCapacity(1000);
        ex.setThreadNamePrefix("notify-");
        ex.initialize();
        return ex;
    }
}
