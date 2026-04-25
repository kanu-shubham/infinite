package com.bookmyshow.payment.repository;

import com.bookmyshow.payment.entity.Payment;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface PaymentRepository extends JpaRepository<Payment, Long> {
    Optional<Payment> findByBookingId(Long bookingId);
    boolean existsByBookingId(Long bookingId);
}
