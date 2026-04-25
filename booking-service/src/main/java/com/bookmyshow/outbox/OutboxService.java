package com.bookmyshow.outbox;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;

/**
 * Helper for callers to write an event to the outbox from inside an existing
 * business transaction. The row is committed together with the business change.
 */
@Service
@RequiredArgsConstructor
public class OutboxService {

    private final OutboxRepository repo;
    private final ObjectMapper mapper;

    /**
     * Requires an active transaction - this method is meant to be called from
     * within a @Transactional business method so the outbox insert is part of
     * the same commit.
     */
    @Transactional(propagation = Propagation.MANDATORY)
    public void save(String topic, String partitionKey, Object payload) {
        try {
            OutboxEvent event = OutboxEvent.builder()
                    .topic(topic)
                    .partitionKey(partitionKey)
                    .payload(mapper.writeValueAsString(payload))
                    .payloadType(payload.getClass().getName())
                    .published(false)
                    .createdAt(Instant.now())
                    .build();
            repo.save(event);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Cannot serialize outbox payload", e);
        }
    }
}
