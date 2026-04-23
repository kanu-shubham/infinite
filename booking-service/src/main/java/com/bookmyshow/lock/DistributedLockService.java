package com.bookmyshow.lock;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Distributed lock built on Redisson.
 *
 * Why Redis / Redisson and not just {@code synchronized}?
 *   - synchronized only works inside one JVM. Run two instances of this service
 *     behind a load balancer and synchronized gives you zero protection.
 *   - Redisson implements the Redlock-style algorithm: SET key value NX PX ttl,
 *     with fencing via a unique lock id and a Lua-scripted release (so thread A
 *     can't accidentally release thread B's lock after A's TTL expires).
 *
 * Always call with a TTL (leaseTime) so a crashed holder does not wedge the lock forever.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DistributedLockService {

    private final RedissonClient redisson;

    /**
     * Acquires multiple locks atomically using Redisson's MultiLock (all-or-nothing).
     * Seats must be acquired in a deterministic order to avoid deadlock - we sort by id.
     */
    public <T> T withSeatLocks(List<Long> seatIds, long waitMs, long leaseMs, Supplier<T> work) {
        List<Long> sorted = new ArrayList<>(seatIds);
        Collections.sort(sorted);

        List<RLock> locks = new ArrayList<>(sorted.size());
        for (Long id : sorted) {
            locks.add(redisson.getLock("seat-lock:" + id));
        }
        RLock multi = redisson.getMultiLock(locks.toArray(new RLock[0]));

        boolean acquired = false;
        try {
            acquired = multi.tryLock(waitMs, leaseMs, TimeUnit.MILLISECONDS);
            if (!acquired) {
                throw new IllegalStateException("Could not acquire seat locks within " + waitMs + "ms");
            }
            return work.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while acquiring seat locks", e);
        } finally {
            if (acquired) {
                try {
                    multi.unlock();
                } catch (Exception e) {
                    // Possible if the lease already expired mid-critical-section. Log and move on.
                    log.warn("Failed to release seat locks {}: {}", sorted, e.getMessage());
                }
            }
        }
    }
}
