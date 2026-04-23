package com.bookmyshow.ratelimit;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Component;

import java.util.Collections;

/**
 * Fixed-window rate limiter backed by a Lua script so INCR + EXPIRE are atomic.
 *
 * Why Lua? Doing {@code INCR key; if first then EXPIRE key ttl} from the client
 * is a race: two clients can see "first" at the same time and only one EXPIRE
 * lands, or worse, both EXPIREs land and reset the window. One Lua script =
 * one atomic operation on the Redis side.
 */
@Component
@RequiredArgsConstructor
public class RedisRateLimiter {

    private final StringRedisTemplate redis;

    private static final DefaultRedisScript<Long> SCRIPT = new DefaultRedisScript<>(
            "local current = redis.call('INCR', KEYS[1]) " +
            "if tonumber(current) == 1 then " +
            "   redis.call('PEXPIRE', KEYS[1], ARGV[1]) " +
            "end " +
            "return current",
            Long.class);

    /** Returns true if the request is allowed, false if the window is saturated. */
    public boolean allow(String bucket, int limit, long windowMillis) {
        Long count = redis.execute(SCRIPT, Collections.singletonList("rl:" + bucket),
                                   String.valueOf(windowMillis));
        return count != null && count <= limit;
    }
}
