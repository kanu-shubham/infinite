import { useState, useCallback, useEffect, useRef } from "react";

export const RATING = Object.freeze({
  NEGATIVE: "NEGATIVE",
  POSITIVE: "POSITIVE",
  STELLAR: "STELLAR",
});

const FLOW = Object.freeze({
  IDLE: "IDLE",
  RATING: "RATING",
  NEGATIVE_FEEDBACK: "NEGATIVE_FEEDBACK",
  THANK_YOU: "THANK_YOU",
  TRUSTPILOT: "TRUSTPILOT",
});

const THANK_YOU_DURATION_MS = 2000;

export default function useFeatureRating() {
  const [flow, setFlow] = useState(FLOW.IDLE);
  const pendingTrustpilot = useRef(false);

  const open = useCallback(() => setFlow(FLOW.RATING), []);
  const close = useCallback(() => {
    pendingTrustpilot.current = false;
    setFlow(FLOW.IDLE);
  }, []);

  const submitRating = useCallback((rating) => {
    if (rating === RATING.NEGATIVE) {
      setFlow(FLOW.NEGATIVE_FEEDBACK);
    } else {
      pendingTrustpilot.current = rating === RATING.STELLAR;
      setFlow(FLOW.THANK_YOU);
    }
  }, []);

  const submitNegativeFeedback = useCallback(() => {
    pendingTrustpilot.current = false;
    setFlow(FLOW.THANK_YOU);
  }, []);

  const dismissTrustpilot = useCallback(() => setFlow(FLOW.IDLE), []);

  useEffect(() => {
    if (flow !== FLOW.THANK_YOU) return;
    const timer = setTimeout(() => {
      if (pendingTrustpilot.current) {
        setFlow(FLOW.TRUSTPILOT);
      } else {
        setFlow(FLOW.IDLE);
      }
    }, THANK_YOU_DURATION_MS);
    return () => clearTimeout(timer);
  }, [flow]);

  return {
    isOpen: flow !== FLOW.IDLE,
    showRating: flow === FLOW.RATING,
    showNegativeFeedback: flow === FLOW.NEGATIVE_FEEDBACK,
    showThankYou: flow === FLOW.THANK_YOU,
    showTrustpilot: flow === FLOW.TRUSTPILOT,
    open,
    close,
    submitRating,
    submitNegativeFeedback,
    dismissTrustpilot,
  };
}
