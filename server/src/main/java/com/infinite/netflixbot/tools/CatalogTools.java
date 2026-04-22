package com.infinite.netflixbot.tools;

import com.infinite.netflixbot.catalog.Show;
import com.infinite.netflixbot.catalog.ShowRepository;
import com.infinite.netflixbot.user.UserProfile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

/**
 * Function-calling tools exposed to the LLM. Spring AI scans @Tool-annotated
 * methods and advertises them as JSON-schema callable functions to Claude.
 *
 * The LLM decides WHEN to call these based on the user message; our code
 * executes them and the result is fed back into the model's context.
 */
@Component
public class CatalogTools {

    private static final Logger log = LoggerFactory.getLogger(CatalogTools.class);

    private final ShowRepository showRepository;
    private final MyListService myList;

    // ThreadLocal user context so tools know who they're acting for.
    // Populated by the agent before each turn; cleared after.
    private static final ThreadLocal<UserProfile> CURRENT = new ThreadLocal<>();

    public CatalogTools(ShowRepository showRepository, MyListService myList) {
        this.showRepository = showRepository;
        this.myList = myList;
    }

    public static void bindUser(UserProfile user) { CURRENT.set(user); }
    public static void clearUser() { CURRENT.remove(); }

    @Tool(description =
        "Returns the top trending shows on Netflix right now. " +
        "Use when the user asks 'what's popular', 'trending', or 'what's everyone watching'.")
    public List<ShowSummary> trending(
            @ToolParam(description = "Max number of shows to return, 1-10", required = false)
            Integer limit) {
        int n = (limit == null || limit <= 0 || limit > 10) ? 5 : limit;
        log.info("tool:trending limit={}", n);
        return showRepository.findTop10ByOrderByPopularityScoreDesc().stream()
                .limit(n)
                .map(ShowSummary::of)
                .toList();
    }

    @Tool(description =
        "Look up the details of a specific show by exact title. " +
        "Use when the user mentions a title and wants info about it.")
    public ShowSummary getShow(
            @ToolParam(description = "Exact show title, case-insensitive") String title) {
        log.info("tool:getShow title={}", title);
        return showRepository.findAll().stream()
                .filter(s -> s.getTitle().equalsIgnoreCase(title))
                .findFirst()
                .map(ShowSummary::of)
                .orElse(null);
    }

    @Tool(description =
        "Returns the current user's recently watched shows. " +
        "Use this before recommending so you can explain what it's similar to.")
    public List<ShowSummary> recentWatches() {
        UserProfile user = CURRENT.get();
        if (user == null) return List.of();
        log.info("tool:recentWatches user={}", user.userId());
        return user.watchedShowIds().stream()
                .map(showRepository::findById)
                .flatMap(Optional::stream)
                .map(ShowSummary::of)
                .toList();
    }

    @Tool(description =
        "Add a show to the user's My List. " +
        "Call this only after the user explicitly says they want to save / add it.")
    public String addToList(
            @ToolParam(description = "Show id, e.g. 'stranger-things'") String showId) {
        UserProfile user = CURRENT.get();
        if (user == null) return "No active user session.";
        if (showRepository.findById(showId).isEmpty()) {
            return "Unknown showId: " + showId;
        }
        myList.add(user.userId(), showId);
        log.info("tool:addToList user={} show={}", user.userId(), showId);
        return "Added " + showId + " to " + user.displayName() + "'s My List.";
    }

    public record ShowSummary(
            String id, String title, int year, String maturity,
            String genres, double imdb, int runtimeMinutes, String synopsis) {
        public static ShowSummary of(Show s) {
            return new ShowSummary(
                    s.getId(), s.getTitle(), s.getYear(), s.getMaturityRating(),
                    s.getGenres(), s.getImdbRating(), s.getRuntimeMinutes(), s.getSynopsis());
        }
    }
}
