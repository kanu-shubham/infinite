package com.bookmyshow.service;

import com.bookmyshow.entity.Movie;
import com.bookmyshow.repository.MovieRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Cache-aside for mostly-static data:
 *   - @Cacheable("movies")        : check cache first, on miss load from DB and store
 *   - @CacheEvict(allEntries)     : bust the cache when the catalog changes
 *
 * Under Spring Cache with Redis, 100 concurrent requests for the same movieId all
 * get served from Redis after the first miss. The underlying DB sees one query, not 100.
 */
@Service
@RequiredArgsConstructor
public class MovieService {

    private final MovieRepository repo;

    @Cacheable(value = "movies", key = "'all'")
    public List<Movie> list() {
        return repo.findAll();
    }

    @Cacheable(value = "movies", key = "#id")
    public Movie get(Long id) {
        return repo.findById(id).orElseThrow(
                () -> new IllegalArgumentException("Movie not found: " + id));
    }

    @CacheEvict(value = "movies", allEntries = true)
    public Movie save(Movie m) {
        return repo.save(m);
    }
}
