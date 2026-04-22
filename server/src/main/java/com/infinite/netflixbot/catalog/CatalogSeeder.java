package com.infinite.netflixbot.catalog;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.InputStream;

/**
 * On startup: load the seed catalog into Postgres and pgvector.
 * Production equivalent: a Spark/Flink batch job that streams from the content
 * catalog service and publishes to Kafka; an indexer service consumes that topic.
 */
@Component
public class CatalogSeeder implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(CatalogSeeder.class);

    private final ShowRepository showRepository;
    private final CatalogIndexer indexer;
    private final ObjectMapper mapper = new ObjectMapper();

    public CatalogSeeder(ShowRepository showRepository, CatalogIndexer indexer) {
        this.showRepository = showRepository;
        this.indexer = indexer;
    }

    @Override
    public void run(String... args) throws Exception {
        if (showRepository.count() > 0) {
            log.info("Catalog already seeded ({} rows) — skipping", showRepository.count());
            return;
        }

        try (InputStream in = new ClassPathResource("data/shows.json").getInputStream()) {
            JsonNode arr = mapper.readTree(in);
            int i = 0;
            for (JsonNode n : arr) {
                Show s = new Show(
                        n.get("id").asText(),
                        n.get("title").asText(),
                        n.get("year").asInt(),
                        n.get("maturity").asText(),
                        n.get("genres").asText(),
                        n.get("cast").asText(),
                        n.get("tags").asText(),
                        n.get("synopsis").asText(),
                        n.get("imdb").asDouble(),
                        n.get("runtime").asInt(),
                        n.get("popularity").asLong()
                );
                showRepository.save(s);
                indexer.index(s);
                i++;
            }
            log.info("Seeded {} shows into Postgres + pgvector", i);
        }
    }
}
