package com.bookmyshow.temporal;

import io.temporal.client.WorkflowClient;
import io.temporal.client.WorkflowOptions;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Tiny REST front-end so you can kick off workflows by curl.
 *
 *   POST /workflows/bookings  -> starts a workflow asynchronously and returns the workflowId.
 *   GET  /workflows/bookings/{id}/result -> blocks until complete, returns the result string.
 */
@RestController
@RequestMapping("/workflows/bookings")
@RequiredArgsConstructor
public class BookingWorkflowController {

    private final WorkflowClient client;

    @PostMapping
    public Map<String, String> start(@RequestBody Map<String, Object> body) {
        String workflowId = "booking-" + UUID.randomUUID();
        BookingWorkflow wf = client.newWorkflowStub(BookingWorkflow.class,
                WorkflowOptions.newBuilder()
                        .setTaskQueue("booking-saga")
                        .setWorkflowId(workflowId)
                        .build());

        @SuppressWarnings("unchecked")
        List<Long> seatIds = ((List<?>) body.get("seatIds")).stream()
                .map(o -> Long.valueOf(o.toString())).toList();

        WorkflowClient.start(wf::runBookingSaga,
                Long.valueOf(body.get("userId").toString()),
                Long.valueOf(body.get("showId").toString()),
                seatIds,
                Long.valueOf(body.getOrDefault("amountCents", "500").toString()));

        return Map.of("workflowId", workflowId);
    }

    @GetMapping("/{id}/result")
    public Map<String, String> result(@PathVariable String id) {
        BookingWorkflow wf = client.newWorkflowStub(BookingWorkflow.class, id);
        // Calling a workflow method on a stub created from an existing workflowId blocks
        // until that workflow completes and returns its result.
        @SuppressWarnings("UnusedDeclaration")
        String r;
        try {
            // We can't easily re-call the @WorkflowMethod with original args; use the untyped stub.
            r = client.newUntypedWorkflowStub(id).getResult(String.class);
        } catch (Exception e) {
            return Map.of("status", "ERROR", "error", e.getMessage());
        }
        return Map.of("status", "DONE", "result", r);
    }
}
