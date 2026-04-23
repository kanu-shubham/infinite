package com.bookmyshow.entity;

import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Table(name = "show")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Show implements Serializable {
    @Id
    private Long id;

    @Column(name = "movie_id")
    private Long movieId;

    private String screen;

    @Column(name = "start_time")
    private LocalDateTime startTime;

    @Column(name = "total_seats")
    private Integer totalSeats;
}
