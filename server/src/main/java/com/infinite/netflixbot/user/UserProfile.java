package com.infinite.netflixbot.user;

import java.util.List;

/**
 * In production this is hydrated from the user service + feature store.
 * The watched list and preferred genres become part of the system prompt
 * so the LLM's recommendations are personalized.
 */
public record UserProfile(
        String userId,
        String displayName,
        String maturityRating,       // parental setting
        String region,
        List<String> preferredGenres,
        List<String> watchedShowIds
) {
    public static UserProfile demo(String userId) {
        return new UserProfile(
                userId,
                "Alex",
                "TV-MA",
                "US",
                List.of("Sci-Fi", "Thriller", "Crime"),
                List.of("dark", "stranger-things", "ozark")
        );
    }
}
