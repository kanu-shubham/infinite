import React from "react";
import Overlay from "./Overlay";
import { RATING } from "../hooks/useFeatureRating";
import "./RatingPopup.css";

const RATING_OPTIONS = [
  { rating: RATING.NEGATIVE, emoji: "👎", label: "Negative" },
  { rating: RATING.POSITIVE, emoji: "👍", label: "Positive" },
  { rating: RATING.STELLAR, emoji: "🤩", label: "Stellar" },
];

export default function RatingPopup({ onRate, onClose }) {
  return (
    <Overlay onClick={onClose}>
      <div className="rating-popup">
        <p className="rating-popup__question">How would you rate this feature?</p>
        <div className="rating-popup__options">
          {RATING_OPTIONS.map(({ rating, emoji, label }) => (
            <button
              key={rating}
              className="rating-popup__option"
              onClick={() => onRate(rating)}
              aria-label={label}
            >
              {emoji}
            </button>
          ))}
        </div>
      </div>
    </Overlay>
  );
}
