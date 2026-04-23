package com.bookmyshow.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.cache.CacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.cache.RedisCacheConfiguration;
import org.springframework.data.redis.cache.RedisCacheManager;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.RedisSerializationContext;
import org.springframework.data.redis.serializer.StringRedisSerializer;

import java.time.Duration;

@Configuration
public class RedisConfig {

    @Bean
    public StringRedisTemplate stringRedisTemplate(RedisConnectionFactory cf) {
        return new StringRedisTemplate(cf);
    }

    /**
     * JSON serialization so cached values are human-readable in redis-cli and survive
     * across application restarts (the default JDK serializer bakes in class paths).
     */
    @Bean
    public CacheManager cacheManager(RedisConnectionFactory cf) {
        RedisCacheConfiguration cfg = RedisCacheConfiguration.defaultCacheConfig()
                .entryTtl(Duration.ofMinutes(1))
                .disableCachingNullValues()
                .serializeKeysWith(RedisSerializationContext.SerializationPair
                        .fromSerializer(new StringRedisSerializer()))
                .serializeValuesWith(RedisSerializationContext.SerializationPair
                        .fromSerializer(new GenericJackson2JsonRedisSerializer(new ObjectMapper())));

        return RedisCacheManager.builder(cf)
                .cacheDefaults(cfg)
                // Per-cache TTLs. Shows change rarely -> 10 min. Seat map changes on every booking -> 5 s.
                .withCacheConfiguration("movies",     cfg.entryTtl(Duration.ofMinutes(10)))
                .withCacheConfiguration("shows",      cfg.entryTtl(Duration.ofMinutes(10)))
                .withCacheConfiguration("seat-map",   cfg.entryTtl(Duration.ofSeconds(5)))
                .build();
    }
}
