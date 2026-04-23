package com.bookmyshow.entity;

import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;

@Entity
@Table(name = "movie")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Movie implements Serializable {
    @Id
    private Long id;

    private String title;
    private String genre;

    @Column(name = "duration_minutes")
    private Integer durationMinutes;
}
