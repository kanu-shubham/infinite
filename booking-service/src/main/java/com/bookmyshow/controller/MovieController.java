package com.bookmyshow.controller;

import com.bookmyshow.entity.Movie;
import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.Show;
import com.bookmyshow.service.MovieService;
import com.bookmyshow.service.ShowService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class MovieController {

    private final MovieService movies;
    private final ShowService  shows;

    @GetMapping("/movies")
    public List<Movie> list() { return movies.list(); }

    @GetMapping("/movies/{id}")
    public Movie get(@PathVariable Long id) { return movies.get(id); }

    @GetMapping("/movies/{id}/shows")
    public List<Show> shows(@PathVariable Long id) { return shows.showsForMovie(id); }

    @GetMapping("/shows/{id}/seats")
    public List<Seat> seats(@PathVariable Long id) { return shows.seatMap(id); }
}
