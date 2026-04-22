package com.infinite.netflixbot.tools;

import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Stand-in for the real "My List" microservice. The recommender agent calls
 * addToList via a tool; the LLM decides when, based on user intent.
 */
@Service
public class MyListService {

    private final ConcurrentHashMap<String, Set<String>> lists = new ConcurrentHashMap<>();

    public void add(String userId, String showId) {
        lists.computeIfAbsent(userId, k -> ConcurrentHashMap.newKeySet()).add(showId);
    }

    public List<String> get(String userId) {
        return List.copyOf(lists.getOrDefault(userId, Set.of()));
    }
}
