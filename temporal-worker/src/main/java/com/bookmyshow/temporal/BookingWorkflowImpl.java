package com.bookmyshow.temporal;

import io.temporal.activity.ActivityOptions;
import io.temporal.common.RetryOptions;
import io.temporal.spring.boot.WorkflowImpl;
import io.temporal.workflow.Workflow;

import java.time.Duration;
import java.util.List;

/**
 * Workflow implementation. Reads as plain sequential Java — Temporal makes it durable.
 *
 * Compare this to BookingSagaCoordinator + OutboxPublisher:
 *   - No outbox table. No Kafka topic. No idempotency tracking. No reaper for stuck sagas.
 *   - Activity retries / backoff are declarative (RetryOptions).
 *   - Compensation is just a try/catch — Temporal guarantees the catch runs even if the worker crashes.
 *   - You can kill the JVM mid-run; restart; the workflow resumes from the next un-completed activity.
 *
 * This is why "real" sagas at scale (Uber, DoorDash, Stripe) use a workflow engine.
 */
@WorkflowImpl(taskQueues = "booking-saga")
public class BookingWorkflowImpl implements BookingWorkflow {

    /**
     * Per-activity options. startToCloseTimeout is the wall-clock budget; retries
     * use exponential backoff. These values are propagated to the worker side.
     */
    private final ActivityOptions defaultOpts = ActivityOptions.newBuilder()
            .setStartToCloseTimeout(Duration.ofSeconds(10))
            .setRetryOptions(RetryOptions.newBuilder()
                    .setInitialInterval(Duration.ofMillis(500))
                    .setMaximumInterval(Duration.ofSeconds(5))
                    .setBackoffCoefficient(2.0)
                    .setMaximumAttempts(5)
                    .build())
            .build();

    private final BookingActivities activities = Workflow.newActivityStub(
            BookingActivities.class, defaultOpts);

    @Override
    public String runBookingSaga(Long userId, Long showId, List<Long> seatIds, long amountCents) {
        Long bookingId = null;
        boolean charged = false;

        try {
            bookingId = activities.holdSeats(userId, showId, seatIds);
            String txnRef = activities.chargePayment(bookingId, userId, amountCents);
            charged = true;
            activities.confirmBooking(bookingId, txnRef);
            activities.sendNotification(bookingId);
            return "CONFIRMED:" + bookingId;
        } catch (Exception e) {
            // Compensate in REVERSE order, only the steps that succeeded.
            if (charged && bookingId != null) {
                activities.refundPayment(bookingId);
            }
            if (bookingId != null) {
                activities.releaseSeats(bookingId);
            }
            // Workflow itself returns a failed-but-handled outcome. The exception was business-level;
            // Temporal will not retry the whole workflow (only activities are retried).
            return "FAILED:" + (bookingId == null ? "?" : bookingId) + ":" + e.getMessage();
        }
    }
}
