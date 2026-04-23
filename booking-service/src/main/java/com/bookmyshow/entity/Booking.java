package com.bookmyshow.entity;

import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Table(name = "booking", indexes = {
        @Index(name = "idx_booking_user", columnList = "user_id"),
        @Index(name = "idx_booking_show", columnList = "show_id"),
        @Index(name = "idx_booking_idem", columnList = "idempotency_key", unique = true)
})
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Booking implements Serializable {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id")
    private Long userId;

    @Column(name = "show_id")
    private Long showId;

    @Column(name = "seat_ids")
    private String seatIds;       // comma-separated; kept flat for simplicity

    @Column(name = "seat_labels")
    private String seatLabels;    // denormalized for display

    @Enumerated(EnumType.STRING)
    private BookingStatus status;

    @Column(name = "idempotency_key", unique = true)
    private String idempotencyKey;

    @Column(name = "created_at")
    private LocalDateTime createdAt;
}
