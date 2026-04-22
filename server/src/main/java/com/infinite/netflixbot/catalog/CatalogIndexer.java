package com.infinite.netflixbot.catalog;

import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

/**
 * Indexes catalog entries into the vector store.
 * In production this is driven by a Kafka consumer on catalog-change events
 * (create / update / delete). Here we call it during startup seeding.
 */
@Service
public class CatalogIndexer {

    private final VectorStore vectorStore;

    public CatalogIndexer(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    public void index(Show show) {
        Document doc = new Document(
                show.getId(),
                show.toEmbeddingText(),
                Map.of(
                        "showId", show.getId(),
                        "title", show.getTitle(),
                        "year", show.getYear(),
                        "maturity", show.getMaturityRating(),
                        "genres", show.getGenres(),
                        "popularity", show.getPopularityScore(),
                        "imdb", show.getImdbRating()
                )
        );
        // VectorStore.add computes the embedding via the configured EmbeddingModel
        // and upserts into pgvector.
        vectorStore.add(List.of(doc));
    }

    public List<Document> retrieveSimilar(String query, int topK, String maturityFilter) {
        SearchRequest req = SearchRequest.builder()
                .query(query)
                .topK(topK)
                .similarityThreshold(0.3)
                .filterExpression(buildMaturityFilter(maturityFilter))
                .build();
        return vectorStore.similaritySearch(req);
    }

    /**
     * Metadata filter so we never leak adult content to under-age profiles.
     * Expression is parsed by Spring AI's FilterExpressionBuilder syntax.
     */
    private String buildMaturityFilter(String maxMaturity) {
        if (maxMaturity == null || maxMaturity.isBlank()) return null;
        List<String> allowed = switch (maxMaturity) {
            case "G"     -> List.of("G");
            case "PG"    -> List.of("G", "PG", "TV-Y", "TV-G", "TV-PG");
            case "PG-13" -> List.of("G", "PG", "PG-13", "TV-Y", "TV-G", "TV-PG", "TV-14");
            default      -> null; // TV-MA -> no filter
        };
        if (allowed == null) return null;
        String list = allowed.stream().map(m -> "'" + m + "'")
                .reduce((a, b) -> a + "," + b).orElse("");
        return "maturity in [" + list + "]";
    }
}
