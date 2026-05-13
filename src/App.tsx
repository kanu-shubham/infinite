import React, { useCallback, useState } from 'react';
import HotelListingPage from './features/hotels/HotelListingPage';
import { FeedbackWidget } from './features/feedback';
import './App.css';

export default function App(): JSX.Element {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const open = useCallback(() => setFeedbackOpen(true), []);
  const close = useCallback(() => setFeedbackOpen(false), []);

  return (
    <div className="app">
      <HotelListingPage />
      <button
        type="button"
        className="app__feedback-fab"
        onClick={open}
        aria-haspopup="dialog"
      >
        Rate this feature
      </button>
      <FeedbackWidget open={feedbackOpen} onClose={close} />
    </div>
  );
}
