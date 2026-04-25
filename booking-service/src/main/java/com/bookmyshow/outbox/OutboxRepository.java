package com.bookmyshow.outbox;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;

public interface OutboxRepository extends JpaRepository<OutboxEvent, Long> {

    /**
     * Returns the oldest un-published events. ORDER BY created_at keeps rough causal order.
     * Pageable limits the batch size - important so we don't OOM on a backlog of 100k events.
     */
    @Query("select o from OutboxEvent o where o.published = false order by o.createdAt asc")
    List<OutboxEvent> findUnpublished(Pageable pageable);
}
