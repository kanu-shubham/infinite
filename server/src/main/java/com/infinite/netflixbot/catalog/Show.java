package com.infinite.netflixbot.catalog;

import jakarta.persistence.*;

@Entity
@Table(name = "shows")
public class Show {

    @Id
    private String id;

    private String title;
    private Integer year;

    @Column(length = 64)
    private String maturityRating; // G, PG, PG-13, TV-MA, etc.

    @Column(length = 256)
    private String genres;         // comma-separated

    @Column(length = 512)
    private String cast;

    @Column(length = 256)
    private String tags;           // moody, slow-burn, etc.

    @Column(length = 4000)
    private String synopsis;

    private Double imdbRating;
    private Integer runtimeMinutes;
    private Long popularityScore; // used for ranking

    public Show() {}

    public Show(String id, String title, int year, String maturity, String genres,
                String cast, String tags, String synopsis, double imdb, int runtime, long popularity) {
        this.id = id;
        this.title = title;
        this.year = year;
        this.maturityRating = maturity;
        this.genres = genres;
        this.cast = cast;
        this.tags = tags;
        this.synopsis = synopsis;
        this.imdbRating = imdb;
        this.runtimeMinutes = runtime;
        this.popularityScore = popularity;
    }

    public String getId() { return id; }
    public String getTitle() { return title; }
    public Integer getYear() { return year; }
    public String getMaturityRating() { return maturityRating; }
    public String getGenres() { return genres; }
    public String getCast() { return cast; }
    public String getTags() { return tags; }
    public String getSynopsis() { return synopsis; }
    public Double getImdbRating() { return imdbRating; }
    public Integer getRuntimeMinutes() { return runtimeMinutes; }
    public Long getPopularityScore() { return popularityScore; }

    /** Flat text used for embedding. */
    public String toEmbeddingText() {
        return """
            %s (%d). Genre: %s. Maturity: %s. Runtime: %d min. IMDb: %.1f.
            Cast: %s.
            Tags: %s.
            Synopsis: %s
            """.formatted(title, year, genres, maturityRating, runtimeMinutes,
                          imdbRating, cast, tags, synopsis);
    }
}
