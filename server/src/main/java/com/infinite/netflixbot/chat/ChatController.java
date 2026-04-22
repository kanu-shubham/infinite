package com.infinite.netflixbot.chat;

import com.infinite.netflixbot.user.UserProfile;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.time.Duration;

/**
 * Real-time streaming chat endpoint.
 *
 * POST /api/chat/stream
 *   body: { sessionId, userId, message }
 *   response: text/event-stream
 *     event: token  -> partial tokens as they arrive from the LLM
 *     event: done   -> final marker
 *     event: error  -> upstream failure
 *
 * SSE over POST works in browsers via the `fetch` Streams API (used by
 * Vercel AI SDK, shadcn chat, etc.). For WebSocket/STOMP, swap this controller
 * for a @MessageMapping handler — the agent logic is identical.
 */
@RestController
@RequestMapping("/api/chat")
@CrossOrigin(origins = "*")
public class ChatController {

    private static final Logger log = LoggerFactory.getLogger(ChatController.class);

    private final RecommenderAgent agent;

    public ChatController(RecommenderAgent agent) {
        this.agent = agent;
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> stream(@Valid @RequestBody ChatRequest req) {
        log.info("chat session={} user={} msg={}", req.sessionId(), req.userId(), req.message());

        // In production: fetch from user service + feature store.
        UserProfile user = UserProfile.demo(req.userId());

        Flux<ServerSentEvent<String>> tokens = agent.reply(req.sessionId(), req.message(), user)
                .map(tok -> ServerSentEvent.<String>builder()
                        .event("token")
                        .data(tok)
                        .build());

        Flux<ServerSentEvent<String>> done = Flux.just(
                ServerSentEvent.<String>builder().event("done").data("").build());

        // Heartbeat keeps proxies from killing the connection on long LLM thinks.
        Flux<ServerSentEvent<String>> heartbeat = Flux.interval(Duration.ofSeconds(15))
                .map(i -> ServerSentEvent.<String>builder().event("ping").data("").build());

        return Flux.merge(tokens.concatWith(done), heartbeat)
                .takeUntil(e -> "done".equals(e.event()))
                .onErrorResume(err -> {
                    log.error("chat stream failed", err);
                    return Flux.just(ServerSentEvent.<String>builder()
                            .event("error")
                            .data(err.getMessage())
                            .build());
                });
    }

    @GetMapping("/healthz")
    public String healthz() { return "ok"; }
}
