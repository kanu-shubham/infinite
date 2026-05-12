import React from 'react';
import './TrustpilotPrompt.css';

const DEFAULT_URL = 'https://www.trustpilot.com/review/bunq.com';

export function TrustpilotPrompt({ trustpilotUrl = DEFAULT_URL, onClose, titleId }) {
  // We could open programmatically, but a real anchor preserves middle-click,
  // ctrl-click, and right-click semantics that users expect.
  return (
    <div className="fb-trustpilot">
      <h2 id={titleId} className="fb-trustpilot__title">Enjoying bunq?</h2>
      <p className="fb-trustpilot__body">
        Your feedback helps us improve!<br />
        Please leave us a review on Trustpilot.
      </p>
      <a
        className="fb-trustpilot__cta"
        href={trustpilotUrl}
        target="_blank"
        rel="noopener noreferrer"
        onClick={() => {
          // Close after the user opts in; the new tab is already open.
          onClose?.();
        }}
        data-testid="fb-trustpilot-cta"
      >
        Go to Trustpilot
      </a>
    </div>
  );
}
