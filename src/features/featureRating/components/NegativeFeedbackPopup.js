import React, { useState } from "react";
import Overlay from "./Overlay";
import "./NegativeFeedbackPopup.css";

export default function NegativeFeedbackPopup({ onSubmit, onClose }) {
  const [feedback, setFeedback] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(feedback);
  }

  return (
    <Overlay onClick={onClose}>
      <div className="negative-feedback-popup">
        <p className="negative-feedback-popup__question">
          How can we make things better?
        </p>
        <form onSubmit={handleSubmit} className="negative-feedback-popup__form">
          <textarea
            className="negative-feedback-popup__textarea"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Tell us what went wrong..."
            rows={4}
            aria-label="Feedback"
          />
          <button
            type="submit"
            className="negative-feedback-popup__submit"
            disabled={!feedback.trim()}
          >
            Submit
          </button>
        </form>
      </div>
    </Overlay>
  );
}
