package com.infinite.netflixbot.chat;

import com.infinite.netflixbot.tools.CatalogTools;
import com.infinite.netflixbot.user.UserProfile;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

import java.util.Map;

/**
 * The orchestrator. Single entry point per chat turn.
 *
 * Pipeline:
 *   1. Hydrate user profile (maturity, prefs, watch history)
 *   2. Build system prompt (persona + personalization context)
 *   3. Attach advisors:
 *        - MessageChatMemoryAdvisor  -> last N turns from ChatMemory
 *        - QuestionAnswerAdvisor     -> embeds user msg, retrieves from pgvector,
 *                                       injects top-K show chunks as grounded context
 *   4. Expose CatalogTools as function-callable tools
 *   5. Stream the response token-by-token via Flux<String>
 */
@Service
public class RecommenderAgent {

    private static final String SYSTEM_PROMPT = """
        You are Netflix Concierge, a warm, concise recommendation bot.

        Rules:
        - Only discuss Netflix titles. If asked off-topic, steer back politely.
        - Ground every recommendation in the <context> shows retrieved below. Never invent titles.
        - When recommending, give 2-3 picks max. For each: title, one-line hook, why it matches the user.
        - Respect the user's maturity rating: %s. Never suggest content above it.
        - Prefer shows the user has NOT already watched: %s.
        - Call tools when you need live data (trending, specific show details, my list).
        - Be conversational; 4 sentences max unless listing picks.

        User: %s, region %s, favorites: %s.
        """;

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final ChatMemory chatMemory;
    private final CatalogTools catalogTools;

    public RecommenderAgent(ChatClient.Builder builder,
                            VectorStore vectorStore,
                            ChatMemory chatMemory,
                            CatalogTools catalogTools) {
        this.chatClient = builder.build();
        this.vectorStore = vectorStore;
        this.chatMemory = chatMemory;
        this.catalogTools = catalogTools;
    }

    public Flux<String> reply(String sessionId, String userMessage, UserProfile user) {
        CatalogTools.bindUser(user);

        String system = SYSTEM_PROMPT.formatted(
                user.maturityRating(),
                user.watchedShowIds(),
                user.displayName(),
                user.region(),
                user.preferredGenres()
        );

        // RAG: embed userMessage, top-8 similar shows, filter by maturity.
        // The advisor rewrites the final prompt to inject <context>...</context>.
        QuestionAnswerAdvisor rag = QuestionAnswerAdvisor.builder(vectorStore)
                .searchRequest(SearchRequest.builder()
                        .topK(8)
                        .similarityThreshold(0.3)
                        .filterExpression(maturityFilter(user.maturityRating()))
                        .build())
                .promptTemplate(new PromptTemplate("""
                        {query}

                        Use the following catalog context to ground your answer.
                        Each entry is a show in our library. Do not recommend titles not listed.

                        <context>
                        {question_answer_context}
                        </context>
                        """))
                .build();

        // Short-term memory: last N turns from Redis/in-memory store.
        MessageChatMemoryAdvisor memory = MessageChatMemoryAdvisor.builder(chatMemory)
                .conversationId(sessionId)
                .build();

        return chatClient.prompt()
                .system(system)
                .user(userMessage)
                .advisors(rag, memory)
                .tools(catalogTools)
                .stream()
                .content()
                .doFinally(sig -> CatalogTools.clearUser());
    }

    private String maturityFilter(String maxMaturity) {
        return switch (maxMaturity) {
            case "PG"    -> "maturity in ['G','PG','TV-Y','TV-G','TV-PG']";
            case "PG-13" -> "maturity in ['G','PG','PG-13','TV-Y','TV-G','TV-PG','TV-14']";
            default      -> null;
        };
    }
}
