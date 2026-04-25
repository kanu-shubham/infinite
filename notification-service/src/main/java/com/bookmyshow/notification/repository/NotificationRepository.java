package com.bookmyshow.notification.repository;

import com.bookmyshow.notification.entity.Notification;
import org.springframework.data.jpa.repository.JpaRepository;

public interface NotificationRepository extends JpaRepository<Notification, Long> {
    boolean existsByBookingId(Long bookingId);
}
