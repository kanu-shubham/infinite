import React, { useCallback, useEffect, useMemo, useReducer, useRef } from 'react';
import { useStableId } from './hooks/useStableId';
import { Modal } from './components/Modal';
import { RatingPrompt } from './components/RatingPrompt';
import { NegativeFeedbackForm } from './components/NegativeFeedbackForm';
import { ThankYouToast } from './components/ThankYouToast';
import { TrustpilotPrompt } from './components/TrustpilotPrompt';
import {
  ACTION,
  RATING,
  STATUS,
  initialState,
  reducer,
} from './state/feedbackMachine';
import { submitFeedback as defaultSubmit } from './services/feedbackService';

/**
 * Top-level orchestrator. Composes the state machine, the modal primitive,
 * and the per-step UI. This file deliberately contains no styling or
 * rendering decisions beyond which step to render — every visual concern
 * lives inside the step component.
 *
 * Props:
 *   open               — controlled "is widget visible" flag.
 *   onClose            — invoked when the widget transitions to CLOSED.
 *   submitFeedback     — DI seam for the network call (test override).
 *   trustpilotUrl      — link target for the STELLAR review CTA.
 *   thankYouDelayMs    — auto-dismiss delay for the thank-you toast.
 */
export function FeedbackWidget({
  open,
  onClose,
  submitFeedback = defaultSubmit,
  trustpilotUrl,
  thankYouDelayMs = 2000,
}) {
  const [state, dispatch] = useReducer(reducer, initialState, (s) => ({
    ...s,
    status: open ? STATUS.RATING : STATUS.CLOSED,
  }));

  // Edge-trigger on `open` prop transitions only. Reacting to every render
  // where `open===true && status===CLOSED` would cause an internal CLOSE
  // (e.g. fired from the auto-dismiss timer) to bounce straight back to
  // RATING before the parent has a chance to flip `open` to false.
  const prevOpenRef = useRef(open);
  useEffect(() => {
    const prev = prevOpenRef.current;
    prevOpenRef.current = open;
    if (open && !prev) dispatch({ type: ACTION.OPEN });
    else if (!open && prev) dispatch({ type: ACTION.CLOSE });
  }, [open]);

  const close = useCallback(() => {
    dispatch({ type: ACTION.CLOSE });
  }, []);

  const handleRate = useCallback((rating) => {
    dispatch({ type: ACTION.RATE, rating });
  }, []);

  const handleSubmit = useCallback(
    async (comment) => {
      dispatch({ type: ACTION.SUBMIT_START });
      try {
        await submitFeedback({ rating: RATING.NEGATIVE, comment });
        dispatch({ type: ACTION.SUBMIT_SUCCESS, comment });
      } catch (err) {
        dispatch({ type: ACTION.SUBMIT_ERROR, error: err?.message || 'Submission failed' });
      }
    },
    [submitFeedback],
  );

  const handleThankYouDone = useCallback(() => {
    dispatch({ type: ACTION.THANK_YOU_DONE });
  }, []);

  // Notify parent exactly once per "non-CLOSED → CLOSED" transition. The
  // wasOpenRef guards against firing on first mount and against firing
  // repeatedly while the parent hasn't yet flipped `open` to false.
  const wasOpenRef = useRef(state.status !== STATUS.CLOSED);
  useEffect(() => {
    if (state.status !== STATUS.CLOSED) {
      wasOpenRef.current = true;
    } else if (wasOpenRef.current) {
      wasOpenRef.current = false;
      onClose?.();
    }
  }, [state.status, onClose]);

  const titleId = useStableId('fb-title');
  const step = state.status;

  const content = useMemo(() => {
    switch (step) {
      case STATUS.RATING:
        return <RatingPrompt onRate={handleRate} titleId={titleId} />;
      case STATUS.NEGATIVE_FORM:
      case STATUS.SUBMITTING:
        return (
          <NegativeFeedbackForm
            onSubmit={handleSubmit}
            submitting={step === STATUS.SUBMITTING}
            error={state.error}
            titleId={titleId}
          />
        );
      case STATUS.THANK_YOU:
        return <ThankYouToast active onDone={handleThankYouDone} delay={thankYouDelayMs} titleId={titleId} />;
      case STATUS.TRUSTPILOT:
        return <TrustpilotPrompt trustpilotUrl={trustpilotUrl} onClose={close} titleId={titleId} />;
      default:
        return null;
    }
  }, [step, state.error, titleId, handleRate, handleSubmit, handleThankYouDone, thankYouDelayMs, trustpilotUrl, close]);

  if (step === STATUS.CLOSED) return null;

  const variant =
    step === STATUS.THANK_YOU ? 'blue'
      : step === STATUS.TRUSTPILOT ? 'review'
      : 'dark';

  // The thank-you toast is informational and shouldn't be dismissable by ESC
  // or backdrop, because doing so would skip the STELLAR → trustpilot step.
  const dismissable = step !== STATUS.THANK_YOU && step !== STATUS.SUBMITTING;

  return (
    <Modal
      open
      onClose={close}
      labelledBy={titleId}
      variant={variant}
      dismissable={dismissable}
      showClose={step !== STATUS.THANK_YOU}
      role={step === STATUS.THANK_YOU ? 'status' : 'dialog'}
    >
      {content}
    </Modal>
  );
}
