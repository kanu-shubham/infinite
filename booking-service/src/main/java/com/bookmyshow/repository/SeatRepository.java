package com.bookmyshow.repository;

import com.bookmyshow.entity.Seat;
import com.bookmyshow.entity.SeatStatus;
import jakarta.persistence.LockModeType;
import jakarta.persistence.QueryHint;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.QueryHints;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface SeatRepository extends JpaRepository<Seat, Long> {

    List<Seat> findByShowId(Long showId);

    /**
     * PESSIMISTIC_WRITE: emits `SELECT ... FOR UPDATE`. The DB locks the rows until the
     * transaction commits; any other transaction trying to lock the same rows blocks.
     * Correct but serializes contenders, so throughput falls under heavy contention.
     *
     * The javax.persistence.lock.timeout hint tells Postgres/MySQL to fail fast instead
     * of blocking forever - essential when 100 threads pile up on the same row.
     */
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @QueryHints({@QueryHint(name = "jakarta.persistence.lock.timeout", value = "3000")})
    @Query("select s from Seat s where s.id in :ids")
    List<Seat> findAllByIdForUpdate(@Param("ids") List<Long> ids);

    /**
     * Conditional update: only flips a seat to HELD if it is currently AVAILABLE.
     * This is an atomic CAS at the SQL level - no Java-side locking needed,
     * and it works across application instances. Returns the number of rows
     * changed; if it's less than the number of requested seats, another booker
     * won at least one seat and we must roll back.
     */
    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query("update Seat s set s.status = :newStatus " +
           "where s.id in :ids and s.status = :expectedStatus")
    int atomicUpdateStatusIfCurrent(@Param("ids") List<Long> ids,
                                    @Param("expectedStatus") SeatStatus expectedStatus,
                                    @Param("newStatus") SeatStatus newStatus);
}
