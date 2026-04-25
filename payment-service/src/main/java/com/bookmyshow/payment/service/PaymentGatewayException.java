package com.bookmyshow.payment.service;

/** Thrown when the (simulated) external payment gateway is unhappy. */
public class PaymentGatewayException extends RuntimeException {
    public PaymentGatewayException(String message) { super(message); }
}
