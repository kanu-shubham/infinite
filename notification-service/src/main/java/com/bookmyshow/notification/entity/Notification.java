package com.bookmyshow.notification.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "notification", indexes = {
        @Index(name = "uk_notification_booking", columnList = "booking_id", unique = true)
})
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Notification {

    public enum Status { SENT, FAILED }

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Unique - one notification per booking. DB-level idempotency. */
    @Column(name = "booking_id", unique = true)
    private Long bookingId;

    private String channel;

    @Enumerated(EnumType.STRING)
    private Status status;

    @Column(name = "sent_at")
    private Instant sentAt;
}
