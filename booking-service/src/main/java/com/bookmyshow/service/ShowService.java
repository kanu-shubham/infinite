package com.bookmyshow.service;

import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.Show;
import com.bookmyshow.repository.SeatRepository;
import com.bookmyshow.repository.ShowRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ShowService {

    private final ShowRepository showRepo;
    private final SeatRepository seatRepo;

    @Cacheable(value = "shows", key = "#movieId")
    public List<Show> showsForMovie(Long movieId) {
        return showRepo.findByMovieId(movieId);
    }

    /** Seat map changes on every booking, so this cache uses a very short TTL (5s).
     *  That's still a huge win: during a seat-picker poll loop, 100 clients hitting
     *  this endpoint only translate to ~1 DB query every 5 seconds. */
    @Cacheable(value = "seat-map", key = "#showId")
    public List<Seat> seatMap(Long showId) {
        return seatRepo.findByShowId(showId);
    }
}
