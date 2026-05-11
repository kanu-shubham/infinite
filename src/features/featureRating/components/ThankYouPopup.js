import React from "react";
import Overlay from "./Overlay";
import "./ThankYouPopup.css";

export default function ThankYouPopup() {
  return (
    <Overlay>
      <div className="thank-you-popup" role="status" aria-live="polite">
        <span className="thank-you-popup__checkmark" aria-hidden="true">✓</span>
        <p className="thank-you-popup__message">Thanks for your feedback!</p>
      </div>
    </Overlay>
  );
}
