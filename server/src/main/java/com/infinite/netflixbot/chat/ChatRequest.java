package com.infinite.netflixbot.chat;

import jakarta.validation.constraints.NotBlank;

public record ChatRequest(
        @NotBlank String sessionId,
        @NotBlank String userId,
        @NotBlank String message
) {}
