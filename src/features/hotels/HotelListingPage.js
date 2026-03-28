import React from "react";
import useHotels from "./hooks/useHotels";
import useInfiniteScroll from "./hooks/useInfiniteScroll";
import HotelFilters from "./components/HotelFilters";
import HotelSort from "./components/HotelSort";
import HotelList from "./components/HotelList";
import LoadingSpinner from "../../components/common/LoadingSpinner";
import ErrorMessage from "../../components/common/ErrorMessage";
import "./HotelListingPage.css";

export default function HotelListingPage() {
  const {
    hotels,
    filters,
    sortBy,
    hasMore,
    isLoading,
    error,
    totalCount,
    updateFilter,
    updateSort,
    loadMore,
    retry,
    resetFilters,
  } = useHotels();

  const sentinelRef = useInfiniteScroll(loadMore, {
    enabled: hasMore && !isLoading && !error,
  });

  const isInitialLoad = isLoading && hotels.length === 0;

  return (
    <div className="hotel-listing">
      <header className="hotel-listing__header">
        <h1 className="hotel-listing__title">Find Your Perfect Stay</h1>
        <p className="hotel-listing__subtitle">
          Browse our curated selection of hotels
        </p>
      </header>

      <HotelFilters
        filters={filters}
        onFilterChange={updateFilter}
        onReset={resetFilters}
      />

      <div className="hotel-listing__toolbar">
        <HotelSort
          value={sortBy}
          onChange={updateSort}
          totalCount={totalCount}
        />
      </div>

      {error && <ErrorMessage message={error} onRetry={retry} />}

      {isInitialLoad && !error && (
        <LoadingSpinner size="large" text="Finding the best hotels for you..." />
      )}

      {!isInitialLoad && !error && (
        <HotelList
          hotels={hotels}
          isLoading={isLoading}
          hasMore={hasMore}
          sentinelRef={sentinelRef}
        />
      )}
    </div>
  );
}
