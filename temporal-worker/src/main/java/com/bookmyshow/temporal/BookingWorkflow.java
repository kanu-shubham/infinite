package com.bookmyshow.temporal;

import io.temporal.workflow.WorkflowInterface;
import io.temporal.workflow.WorkflowMethod;

import java.util.List;

/**
 * The workflow contract — a Java interface that Temporal turns into a durable execution.
 *
 * What "durable" means in Temporal:
 *   - Every step the workflow code takes (calls to activities, timers, signals) is persisted.
 *   - If the worker process crashes, Temporal replays the persisted history on a new worker
 *     and the workflow resumes from exactly where it left off.
 *   - The code below is plain Java, but reads as if there were no failures or restarts.
 *     Temporal solves what we wrote by hand in BookingSagaCoordinator + outbox.
 */
@WorkflowInterface
public interface BookingWorkflow {

    @WorkflowMethod
    String runBookingSaga(Long userId, Long showId, List<Long> seatIds, long amountCents);
}
