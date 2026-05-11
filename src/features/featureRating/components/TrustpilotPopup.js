import React from "react";
import Overlay from "./Overlay";
import "./TrustpilotPopup.css";

const TRUSTPILOT_URL = "https://www.trustpilot.com/review/www.bunq.com";

export default function TrustpilotPopup({ onClose }) {
  return (
    <Overlay onClick={onClose}>
      <div className="trustpilot-popup">
        <button
          className="trustpilot-popup__close"
          onClick={onClose}
          aria-label="Close"
        >
          ✕
        </button>
        <p className="trustpilot-popup__title">Enjoying bunq?</p>
        <p className="trustpilot-popup__body">
          Your feedback helps us improve!
          <br />
          Please leave us a review on Trustpilot.
        </p>
        <a
          className="trustpilot-popup__cta"
          href={TRUSTPILOT_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Go to Trustpilot
        </a>
      </div>
    </Overlay>
  );
}
