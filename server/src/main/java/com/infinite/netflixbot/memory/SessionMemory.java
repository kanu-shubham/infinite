package com.infinite.netflixbot.memory;

import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Short-term memory: last N turns per session.
 *
 * Production setup:
 *   - Redis-backed ChatMemoryRepository (TTL 24h)
 *   - Postgres for mid-term summaries (rolled up nightly)
 *   - Vector store for long-term user preference embeddings
 */
@Configuration
public class SessionMemory {

    @Bean
    public ChatMemory chatMemory() {
        return MessageWindowChatMemory.builder()
                .chatMemoryRepository(new InMemoryChatMemoryRepository())
                .maxMessages(20)
                .build();
    }
}
