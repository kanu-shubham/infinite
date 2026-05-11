import React from "react";
import useFeatureRating from "./hooks/useFeatureRating";
import RatingPopup from "./components/RatingPopup";
import ThankYouPopup from "./components/ThankYouPopup";
import NegativeFeedbackPopup from "./components/NegativeFeedbackPopup";
import TrustpilotPopup from "./components/TrustpilotPopup";
import "./FeatureRating.css";

export default function FeatureRating() {
  const {
    showRating,
    showNegativeFeedback,
    showThankYou,
    showTrustpilot,
    open,
    close,
    submitRating,
    submitNegativeFeedback,
    dismissTrustpilot,
  } = useFeatureRating();

  return (
    <>
      <button className="feature-rating__trigger" onClick={open}>
        Rate this feature
      </button>

      {showRating && (
        <RatingPopup onRate={submitRating} onClose={close} />
      )}

      {showNegativeFeedback && (
        <NegativeFeedbackPopup
          onSubmit={submitNegativeFeedback}
          onClose={close}
        />
      )}

      {showThankYou && <ThankYouPopup />}

      {showTrustpilot && <TrustpilotPopup onClose={dismissTrustpilot} />}
    </>
  );
}
