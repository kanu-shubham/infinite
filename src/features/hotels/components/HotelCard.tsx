import React, { memo } from 'react';
import type { Hotel } from '../types';
import './HotelCard.css';

interface StarRatingProps {
  rating: number;
}

const StarRating = memo(function StarRating({ rating }: StarRatingProps): JSX.Element {
  const fullStars = Math.floor(rating);
  const hasHalf = rating - fullStars >= 0.5;
  const emptyStars = 5 - fullStars - (hasHalf ? 1 : 0);

  return (
    <span className="star-rating" role="img" aria-label={`${rating} out of 5 stars`}>
      <span aria-hidden="true">
        {'★'.repeat(fullStars)}
        {hasHalf && '½'}
        {'☆'.repeat(emptyStars)}
      </span>
    </span>
  );
});

export interface HotelCardProps {
  hotel: Hotel;
}

const HotelCard = memo(function HotelCard({ hotel }: HotelCardProps): JSX.Element {
  return (
    <article
      className="hotel-card"
      aria-label={`${hotel.name}, ${hotel.location}, $${hotel.price} per night, rated ${hotel.rating} out of 5`}
    >
      <div className="hotel-card__image-wrapper">
        <img
          className="hotel-card__image"
          src={hotel.image}
          alt={`Exterior view of ${hotel.name} in ${hotel.location}`}
          loading="lazy"
        />
        <span className="hotel-card__price" aria-hidden="true">
          ${hotel.price}<small>/night</small>
        </span>
      </div>
      <div className="hotel-card__body">
        <h3 className="hotel-card__name">{hotel.name}</h3>
        <p className="hotel-card__location">{hotel.location}</p>
        <div className="hotel-card__rating">
          <StarRating rating={hotel.rating} />
          <span className="hotel-card__rating-value">{hotel.rating}</span>
          <span className="hotel-card__reviews">({hotel.reviewCount} reviews)</span>
        </div>
        <div className="hotel-card__amenities">
          {hotel.amenities.slice(0, 4).map((amenity) => (
            <span key={amenity} className="hotel-card__amenity">{amenity}</span>
          ))}
          {hotel.amenities.length > 4 && (
            <span className="hotel-card__amenity hotel-card__amenity--more">
              +{hotel.amenities.length - 4} more
            </span>
          )}
        </div>
      </div>
    </article>
  );
});

export default HotelCard;
