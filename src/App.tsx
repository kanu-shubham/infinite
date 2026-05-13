import React, { useCallback, useState } from 'react';
import { FeedbackWidget } from './features/feedback';
import './App.css';

export default function App(): JSX.Element {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const open = useCallback(() => setFeedbackOpen(true), []);
  const close = useCallback(() => setFeedbackOpen(false), []);

  return (
    <div className="app">
      <main className="app__shell">
        <h1 className="app__title">Feature Rating Demo</h1>
        <p className="app__lede">
          Click the button below to open the feedback popup.
        </p>
        <button
          type="button"
          className="app__cta"
          onClick={open}
          aria-haspopup="dialog"
        >
          How would you rate this feature?
        </button>
      </main>
      <FeedbackWidget open={feedbackOpen} onClose={close} />
    </div>
  );
}
