import React, { useState } from 'react';
import useHotelFilters from './hooks/useHotelFilters';
import useHotels from './hooks/useHotels';
import useAllHotels from './hooks/useAllHotels';
import useInfiniteScroll from './hooks/useInfiniteScroll';
import HotelFilters from './components/HotelFilters';
import HotelSort from './components/HotelSort';
import HotelList from './components/HotelList';
import VirtualHotelList, { VirtualStrategy } from './components/VirtualHotelList';
import ErrorMessage from '../../components/common/ErrorMessage';
import './HotelListingPage.css';

type TabId = 'virtual' | 'standard';

interface Tab {
  id: TabId;
  label: string;
}

interface Strategy {
  id: VirtualStrategy;
  label: string;
}

const TABS: ReadonlyArray<Tab> = [
  { id: 'virtual',  label: 'Virtualized' },
  { id: 'standard', label: 'Standard'    },
];

const STRATEGIES: ReadonlyArray<Strategy> = [
  { id: 'container', label: 'Container Scroll' },
  { id: 'window',    label: 'Window Scroll'    },
];

export default function HotelListingPage(): JSX.Element {
  const [activeTab, setActiveTab] = useState<TabId>('virtual');
  const [strategy, setStrategy] = useState<VirtualStrategy>('container');

  const {
    rawFilters,
    filters,
    sortBy,
    hasActiveFilters,
    updateFilter,
    updateSort,
    resetFilters,
  } = useHotelFilters();

  const {
    hotels: allHotels,
    isLoading: allLoading,
    error: allError,
    retry: allRetry,
  } = useAllHotels({ filters, sortBy });

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
      <header className="hotel-listing__header">
        <h1 className="hotel-listing__title">Find Your Perfect Stay</h1>
        <p className="hotel-listing__subtitle">
          Browse our curated selection of hotels
        </p>
      </header>

      <HotelFilters
        filters={rawFilters}
        onFilterChange={updateFilter}
        onReset={resetFilters}
        hasActiveFilters={hasActiveFilters}
      />

      <div className="hotel-listing__toolbar">
        <div className="tab-bar">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-bar__btn${activeTab === tab.id ? ' tab-bar__btn--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <HotelSort
          value={sortBy}
          onChange={updateSort}
          totalCount={activeTab === 'virtual' ? allHotels.length : totalCount}
        />
      </div>

      {activeTab === 'virtual' && (
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
                  className={`strategy-toggle__btn${strategy === s.id ? ' strategy-toggle__btn--active' : ''}`}
                  onClick={() => setStrategy(s.id)}
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

      {activeTab === 'standard' && (
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
