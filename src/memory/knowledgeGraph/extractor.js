import { tokenize } from "../utils";

// Heuristic entity/relation extractor tailored to short domain texts like
// hotel descriptions, search phrases, or user actions. Production consumers
// should register a richer extractor (LLM, NER model) via MemoryManager
// options — this version is pluggable, not final.

const DEFAULT_AMENITIES = new Set([
  "pool", "spa", "gym", "sauna", "wifi", "breakfast", "parking",
  "bar", "restaurant", "beach", "kids", "pet", "pets", "shuttle",
  "airport", "concierge", "laundry", "balcony", "view", "sea",
  "ocean", "mountain", "jacuzzi", "fitness",
]);

const DEFAULT_LOCATION_HINTS = new Set([
  "paris", "london", "tokyo", "rome", "berlin", "barcelona", "madrid",
  "amsterdam", "prague", "lisbon", "dublin", "vienna", "athens",
  "istanbul", "dubai", "bangkok", "singapore", "sydney", "melbourne",
  "newyork", "sanfrancisco", "chicago", "miami", "boston", "seattle",
]);

/**
 * @param {{
 *   amenities?: Iterable<string>,
 *   locations?: Iterable<string>,
 *   minTokenLen?: number
 * }} [options]
 */
export function createHeuristicExtractor(options = {}) {
  const amenities = new Set(options.amenities || DEFAULT_AMENITIES);
  const locations = new Set(options.locations || DEFAULT_LOCATION_HINTS);
  const minTokenLen = options.minTokenLen || 3;

  return {
    name: "heuristic",
    /**
     * @param {import("../types").Episode | { text: string, meta?: any, tags?: string[] }} episode
     */
    extract(episode) {
      const text = episode && episode.text ? String(episode.text) : "";
      const meta = (episode && episode.meta) || {};
      const tokens = tokenize(text);
      const nodes = [];
      const edges = [];

      const subjectId = meta.subjectId || (episode && episode.id) || null;
      const subjectLabel = meta.subjectLabel || meta.hotelName || text.slice(0, 40);
      const subjectType = meta.subjectType || "concept";

      if (subjectId) {
        nodes.push({
          id: subjectId,
          type: subjectType,
          label: subjectLabel,
          props: meta.subjectProps || {},
          weight: 1,
        });
      }

      const seenAmenities = new Set();
      const seenLocations = new Set();
      for (const tok of tokens) {
        if (tok.length < minTokenLen) continue;
        if (amenities.has(tok) && !seenAmenities.has(tok)) {
          seenAmenities.add(tok);
          const amenityId = `amenity:${tok}`;
          nodes.push({ id: amenityId, type: "amenity", label: tok });
          if (subjectId) {
            edges.push({
              from: subjectId,
              to: amenityId,
              relation: "has_amenity",
              weight: 1,
              props: { evidence: episode.id || null },
            });
          }
        }
        if (locations.has(tok) && !seenLocations.has(tok)) {
          seenLocations.add(tok);
          const locId = `location:${tok}`;
          nodes.push({ id: locId, type: "location", label: tok });
          if (subjectId) {
            edges.push({
              from: subjectId,
              to: locId,
              relation: "located_in",
              weight: 1,
              props: { evidence: episode.id || null },
            });
          }
        }
      }

      if (Array.isArray(meta.amenities)) {
        for (const a of meta.amenities) {
          const tok = String(a).toLowerCase();
          const amenityId = `amenity:${tok}`;
          nodes.push({ id: amenityId, type: "amenity", label: tok });
          if (subjectId) {
            edges.push({
              from: subjectId,
              to: amenityId,
              relation: "has_amenity",
              weight: 1.5,
              props: { evidence: episode.id || null },
            });
          }
        }
      }
      if (meta.location) {
        const tok = String(meta.location).toLowerCase().replace(/\s+/g, "_");
        const locId = `location:${tok}`;
        nodes.push({ id: locId, type: "location", label: meta.location });
        if (subjectId) {
          edges.push({
            from: subjectId,
            to: locId,
            relation: "located_in",
            weight: 1.5,
            props: { evidence: episode.id || null },
          });
        }
      }

      return { nodes, edges };
    },
  };
}

export default createHeuristicExtractor;
