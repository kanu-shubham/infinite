package com.infinite.netflixbot.catalog;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ShowRepository extends JpaRepository<Show, String> {
    List<Show> findTop10ByOrderByPopularityScoreDesc();
}
