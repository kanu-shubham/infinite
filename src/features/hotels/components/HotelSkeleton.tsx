import React from 'react';
import './HotelSkeleton.css';

function HotelSkeletonCard(): JSX.Element {
  return (
    <div className="hotel-skeleton" aria-hidden="true">
      <div className="hotel-skeleton__image" />
      <div className="hotel-skeleton__body">
        <div className="hotel-skeleton__line hotel-skeleton__line--title" />
        <div className="hotel-skeleton__line hotel-skeleton__line--subtitle" />
        <div className="hotel-skeleton__rating">
          <div className="hotel-skeleton__line hotel-skeleton__line--stars" />
          <div className="hotel-skeleton__line hotel-skeleton__line--reviews" />
        </div>
        <div className="hotel-skeleton__amenities">
          {[1, 2, 3].map((i) => (
            <div key={i} className="hotel-skeleton__pill" />
          ))}
        </div>
      </div>
    </div>
  );
}

export interface HotelSkeletonGridProps {
  count?: number;
}

export default function HotelSkeletonGrid({ count = 8 }: HotelSkeletonGridProps): JSX.Element {
  return (
    <div className="hotel-skeleton-grid" role="status" aria-label="Loading hotels…">
      {Array.from({ length: count }, (_, i) => (
        <HotelSkeletonCard key={i} />
      ))}
    </div>
  );
}
