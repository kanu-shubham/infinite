import { useCallback, useEffect, useRef } from "react";
import { useMemory } from "../../../memory";

// Translates hotel-browsing actions into memory signals:
//   - filter commits → short-term context entries (low salience)
//   - non-default searches → persisted long-term episodes (high salience)
//   - hotel Save / Hide → LTM episode + feedback signal into the retriever

function serializeFilters(filters, sortBy) {
  const parts = [];
  if (filters.search) parts.push(`search: ${filters.search}`);
  if (filters.priceRange) parts.push(`price: ${filters.priceRange}`);
  if (filters.minRating) parts.push(`min-rating: ${filters.minRating}`);
  if (sortBy) parts.push(`sort: ${sortBy}`);
  return parts.length ? parts.join("; ") : "browse: default";
}

export default function useHotelMemory({ filters, sortBy, hasActiveFilters }) {
  const manager = useMemory();
  const lastKeyRef = useRef(null);

  useEffect(() => {
    const text = serializeFilters(filters, sortBy);
    if (text === lastKeyRef.current) return;
    lastKeyRef.current = text;
    manager
      .observe({
        text,
        role: "user",
        salience: hasActiveFilters ? 2 : 0.8,
        persist: Boolean(hasActiveFilters && filters.search),
        tags: ["filter"],
        meta: { kind: "filter", filters, sortBy },
      })
      .catch(() => {});
  }, [manager, filters, sortBy, hasActiveFilters]);

  const saveHotel = useCallback(
    (hotel) => {
      const text = `${hotel.name} — ${hotel.location} — ${(hotel.amenities || []).join(", ")}`;
      return manager.ltm
        .remember({
          id: `hotel:${hotel.id}`,
          text,
          tags: ["hotel", "saved"],
          strength: 2,
          meta: {
            kind: "hotel",
            hotelId: hotel.id,
            subjectId: `hotel:${hotel.id}`,
            subjectType: "hotel",
            subjectLabel: hotel.name,
            location: hotel.location,
            amenities: hotel.amenities,
            price: hotel.price,
            rating: hotel.rating,
          },
        })
        .then((ep) => {
          manager.feedback({ resultId: ep.id, reward: 1 });
          return ep;
        });
    },
    [manager]
  );

  const hideHotel = useCallback(
    (hotel) => {
      const id = `hotel:${hotel.id}`;
      manager.feedback({ resultId: id, reward: -1 });
      return manager.ltm.forget(id);
    },
    [manager]
  );

  return { saveHotel, hideHotel };
}
