package com.bookmyshow.events;

/** Saga topic names used by all services. Keep these in sync across services. */
public final class SagaTopics {
    private SagaTopics() {}

    public static final String SEATS_RESERVED      = "saga.seats.reserved";
    public static final String PAYMENT_CHARGED     = "saga.payment.charged";
    public static final String PAYMENT_FAILED      = "saga.payment.failed";
    public static final String NOTIFICATION_SENT   = "saga.notification.sent";
    public static final String BOOKING_COMPENSATE  = "saga.booking.compensate";
}
