import React from 'react';
import HotelCard from './HotelCard';
import HotelSkeletonGrid from './HotelSkeleton';
import LoadingSpinner from '../../../components/common/LoadingSpinner';
import type { Hotel } from '../types';
import './HotelList.css';

export interface HotelListProps {
  hotels: Hotel[];
  isLoading: boolean;
  hasMore: boolean;
  sentinelRef: (node: Element | null) => void;
}

export default function HotelList({
  hotels,
  isLoading,
  hasMore,
  sentinelRef,
}: HotelListProps): JSX.Element {
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
    <div className="hotel-list" aria-busy={isLoading}>
      <ul className="hotel-list__grid" role="list">
        {hotels.map((hotel) => (
          <li key={hotel.id}>
            <HotelCard hotel={hotel} />
          </li>
        ))}
      </ul>

      {hasMore && (
        <div
          ref={sentinelRef as unknown as React.Ref<HTMLDivElement>}
          className="hotel-list__sentinel"
          aria-hidden="true"
        >
          <LoadingSpinner size="small" text="Loading more hotels..." />
        </div>
      )}

      {!hasMore && hotels.length > 0 && (
        <p className="hotel-list__end" role="status">You've seen all results</p>
      )}
    </div>
  );
}
