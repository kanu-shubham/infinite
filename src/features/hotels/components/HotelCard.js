import React, { memo } from "react";
import "./HotelCard.css";

// memo: hotel objects are stable references from the service layer.
// Without memo, every loadMore call re-renders all existing cards because
// the parent array reference changes ([...prev, ...newItems]).
const StarRating = memo(function StarRating({ rating }) {
  const fullStars = Math.floor(rating);
  const hasHalf = rating - fullStars >= 0.5;
  const emptyStars = 5 - fullStars - (hasHalf ? 1 : 0);

  return (
    <span className="star-rating" aria-label={`${rating} out of 5 stars`}>
      {"★".repeat(fullStars)}
      {hasHalf && "½"}
      {"☆".repeat(emptyStars)}
    </span>
  );
});

const HotelCard = memo(function HotelCard({ hotel, onSave, onHide }) {
  return (
    <article className="hotel-card">
      <div className="hotel-card__image-wrapper">
        <img
          className="hotel-card__image"
          src={hotel.image}
          alt={hotel.name}
          loading="lazy"
        />
        <span className="hotel-card__price">${hotel.price}<small>/night</small></span>
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
        {(onSave || onHide) && (
          <div className="hotel-card__actions">
            {onSave && (
              <button
                type="button"
                className="hotel-card__action hotel-card__action--save"
                onClick={() => onSave(hotel)}
                aria-label={`Save ${hotel.name} to memory`}
              >
                Save
              </button>
            )}
            {onHide && (
              <button
                type="button"
                className="hotel-card__action hotel-card__action--hide"
                onClick={() => onHide(hotel)}
                aria-label={`Hide ${hotel.name}`}
              >
                Hide
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  );
});

export default HotelCard;
