import React, { useState, useCallback } from "react";
import useHotelFilters from "./hooks/useHotelFilters";
import useHotels from "./hooks/useHotels";
import useAllHotels from "./hooks/useAllHotels";
import useInfiniteScroll from "./hooks/useInfiniteScroll";
import HotelFilters from "./components/HotelFilters";
import HotelSort from "./components/HotelSort";
import HotelList from "./components/HotelList";
import VirtualHotelList from "./components/VirtualHotelList";
import ErrorMessage from "../../components/common/ErrorMessage";
import "./HotelListingPage.css";

const TABS = [
  { id: "virtual", label: "Virtualized" },
  { id: "standard", label: "Standard" },
];

const STRATEGIES = [
  { id: "container", label: "Container Scroll" },
  { id: "window",    label: "Window Scroll" },
];

export default function HotelListingPage() {
  const [activeTab, setActiveTab]         = useState("virtual");
  const [strategy, setStrategy]           = useState("container");

  const handleTabChange      = useCallback((id) => setActiveTab(id), []);
  const handleStrategyChange = useCallback((id) => setStrategy(id), []);

  // ── Shared filter + sort state (debounced search inside) ──────────────
  const {
    rawFilters,
    filters,
    sortBy,
    hasActiveFilters,
    updateFilter,
    updateSort,
    resetFilters,
  } = useHotelFilters();

  // ── Tab A: all hotels → virtualized rendering ─────────────────────────
  const {
    hotels: allHotels,
    isLoading: allLoading,
    error: allError,
    retry: allRetry,
  } = useAllHotels({ filters, sortBy });

  // ── Tab B: paginated hotels → infinite scroll ─────────────────────────
  const {
    hotels,
    hasMore,
    isLoading,
    error,
    totalCount,
    loadMore,
    retry,
  } = useHotels({ filters, sortBy });

  const sentinelRef = useInfiniteScroll(loadMore, {
    enabled: hasMore && !isLoading && !error,
  });

  return (
    <div className="hotel-listing">
      {/* ── Header ── */}
      <header className="hotel-listing__header">
        <h1 className="hotel-listing__title">Find Your Perfect Stay</h1>
        <p className="hotel-listing__subtitle">
          Browse our curated selection of hotels
        </p>
      </header>

      {/* ── Shared filters ── */}
      <HotelFilters
        filters={rawFilters}
        onFilterChange={updateFilter}
        onReset={resetFilters}
        hasActiveFilters={hasActiveFilters}
      />

      {/* ── Toolbar: tabs + sort ── */}
      <div className="hotel-listing__toolbar">
        <div className="tab-bar">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-bar__btn${activeTab === tab.id ? " tab-bar__btn--active" : ""}`}
              onClick={() => handleTabChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <HotelSort
          value={sortBy}
          onChange={updateSort}
          totalCount={activeTab === "virtual" ? allHotels.length : totalCount}
        />
      </div>

      {/* ── Virtualized tab ── */}
      {activeTab === "virtual" && (
        <section className="hotel-listing__section">
          <div className="hotel-listing__section-header">
            <h2 className="hotel-listing__section-title">
              Virtualized List
              <span className="hotel-listing__section-subtitle">
                All matching hotels loaded; only visible rows rendered
              </span>
            </h2>
            <div className="strategy-toggle">
              {STRATEGIES.map((s) => (
                <button
                  key={s.id}
                  className={`strategy-toggle__btn${strategy === s.id ? " strategy-toggle__btn--active" : ""}`}
                  onClick={() => handleStrategyChange(s.id)}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <VirtualHotelList
            hotels={allHotels}
            isLoading={allLoading}
            error={allError}
            retry={allRetry}
            strategy={strategy}
          />
        </section>
      )}

      {/* ── Standard tab ── */}
      {activeTab === "standard" && (
        <section className="hotel-listing__section">
          <div className="hotel-listing__section-header">
            <h2 className="hotel-listing__section-title">
              Standard List
              <span className="hotel-listing__section-subtitle">
                Paginated infinite scroll — DOM grows as you scroll
              </span>
            </h2>
          </div>

          {error && <ErrorMessage message={error} onRetry={retry} />}

          {!error && (
            <HotelList
              hotels={hotels}
              isLoading={isLoading}
              hasMore={hasMore}
              sentinelRef={sentinelRef}
            />
          )}
        </section>
      )}
    </div>
  );
}
