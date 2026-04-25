package simulations;

import io.gatling.javaapi.core.ScenarioBuilder;
import io.gatling.javaapi.core.Simulation;
import io.gatling.javaapi.http.HttpProtocolBuilder;

import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.ThreadLocalRandom;

import static io.gatling.javaapi.core.CoreDsl.*;
import static io.gatling.javaapi.http.HttpDsl.*;

/**
 * 1000 rps sustained for 10 minutes against the booking-service.
 *
 * Run via:
 *   mvn -pl gatling-loadtest gatling:test \
 *       -Dgatling.simulationClass=simulations.BookingSimulation
 *
 * Tunables (system properties):
 *   -DbaseUrl=http://localhost:8080         (or your nginx URL: http://localhost:8090)
 *   -Drps=1000                              (target requests/sec)
 *   -Dduration=600                          (total seconds)
 *   -DshowId=1
 *   -DseatRange=20                          (random seat in 1..N)
 *
 * Reads /api/movies (cached, cheap) AND POSTs /api/bookings (the contended path),
 * giving you a realistic mixed load — caching hits AND seat contention together.
 */
public class BookingSimulation extends Simulation {

    private static final String BASE_URL  = System.getProperty("baseUrl", "http://localhost:8080");
    private static final int    RPS       = Integer.parseInt(System.getProperty("rps", "1000"));
    private static final int    DURATION  = Integer.parseInt(System.getProperty("duration", "600"));
    private static final long   SHOW_ID   = Long.parseLong(System.getProperty("showId", "1"));
    private static final int    SEAT_RANGE = Integer.parseInt(System.getProperty("seatRange", "20"));

    private final HttpProtocolBuilder httpProtocol = http
            .baseUrl(BASE_URL)
            .acceptHeader("application/json")
            .contentTypeHeader("application/json")
            .userAgentHeader("gatling-bookmyshow");

    /** Read path: hits the cache after the first request. */
    private final ScenarioBuilder browse = scenario("browse")
            .exec(http("list-movies").get("/api/movies"))
            .exec(http("show-seats").get("/api/shows/" + SHOW_ID + "/seats"));

    /** Write path: the contended booking. Uses optimistic strategy by default. */
    private final ScenarioBuilder book = scenario("book")
            .exec(session -> {
                long userId = ThreadLocalRandom.current().nextLong(1, 1_000_000);
                long seat   = ThreadLocalRandom.current().nextLong(1, SEAT_RANGE + 1);
                return session
                        .set("userId", userId)
                        .set("seat",   seat)
                        .set("idem",   UUID.randomUUID().toString());
            })
            .exec(http("book-seat")
                    .post("/api/bookings")
                    .header("Idempotency-Key", "#{idem}")
                    .body(StringBody("""
                            {"userId": #{userId}, "showId": %d, "seatIds": [#{seat}]}
                            """.formatted(SHOW_ID)))
                    .check(status().in(200, 409, 429)));   // 409=seat taken, 429=rate-limited

    {
        // 80% reads, 20% writes is realistic for ticketing platforms.
        setUp(
                browse.injectOpen(constantUsersPerSec((int)(RPS * 0.8))
                        .during(Duration.ofSeconds(DURATION))),
                book  .injectOpen(constantUsersPerSec((int)(RPS * 0.2))
                        .during(Duration.ofSeconds(DURATION)))
        ).protocols(httpProtocol)
         .assertions(
                global().responseTime().percentile3().lt(500),  // p95 < 500ms
                global().responseTime().percentile4().lt(2000), // p99 < 2s
                global().failedRequests().percent().lt(1.0)     // <1% errors
         );
    }
}
