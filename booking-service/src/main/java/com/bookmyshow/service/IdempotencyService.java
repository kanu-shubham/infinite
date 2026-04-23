package com.bookmyshow.service;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;

/**
 * Idempotency guard.
 *
 * Clients send an `Idempotency-Key` header; we SETNX it in Redis with a TTL.
 * If SETNX returns false the request is a retry - the caller should not create a
 * second booking. This is the standard pattern used by Stripe/Square.
 *
 * The persisted Booking row also has a unique index on idempotency_key as a
 * belt-and-braces guard - if Redis is unavailable we still rely on the DB constraint.
 */
@Service
@RequiredArgsConstructor
public class IdempotencyService {

    private final StringRedisTemplate redis;

    public boolean tryClaim(String key) {
        if (key == null || key.isBlank()) return true; // no key = no guard
        Boolean ok = redis.opsForValue().setIfAbsent("idem:" + key, "1", Duration.ofHours(24));
        return Boolean.TRUE.equals(ok);
    }
}
