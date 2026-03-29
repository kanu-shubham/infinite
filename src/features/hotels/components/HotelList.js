import React from "react";
import HotelCard from "./HotelCard";
import HotelSkeletonGrid from "./HotelSkeleton";
import LoadingSpinner from "../../../components/common/LoadingSpinner";
import "./HotelList.css";

/**
 * @param {boolean} isPaginated
 *   false → shows IntersectionObserver sentinel at the bottom for infinite scroll
 *   true  → hides the sentinel; caller renders <Pagination> below this component
 */
export default function HotelList({ hotels, isLoading, hasMore, sentinelRef, isPaginated = false }) {
  // Initial load — no hotels yet, show skeleton grid instead of spinner
  if (isLoading && hotels.length === 0) {
    return <HotelSkeletonGrid count={8} />;
  }

  if (!isLoading && hotels.length === 0) {
    return (
      <div className="hotel-list__empty">
        <p className="hotel-list__empty-title">No hotels found</p>
        <p className="hotel-list__empty-text">
          Try adjusting your filters to see more results.
        </p>
      </div>
    );
  }

  return (
    <div className="hotel-list">
      <div className="hotel-list__grid">
        {hotels.map((hotel) => (
          <HotelCard key={hotel.id} hotel={hotel} />
        ))}
      </div>

      {/* Infinite scroll sentinel — hidden when using traditional pagination */}
      {!isPaginated && hasMore && (
        <div ref={sentinelRef} className="hotel-list__sentinel">
          <LoadingSpinner size="small" text="Loading more hotels..." />
        </div>
      )}

      {/* "End of results" only makes sense for infinite scroll */}
      {!isPaginated && !hasMore && hotels.length > 0 && (
        <p className="hotel-list__end">You've seen all results</p>
      )}
    </div>
  );
}
