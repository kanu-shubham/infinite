import React from "react";
import "./HotelCard.css";

function StarRating({ rating }) {
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
}

export default function HotelCard({ hotel }) {
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
      </div>
    </article>
  );
}
