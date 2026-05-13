import React from 'react';
import './LoadingSpinner.css';

export type SpinnerSize = 'small' | 'medium' | 'large';

export interface LoadingSpinnerProps {
  size?: SpinnerSize;
  text?: string;
}

export default function LoadingSpinner({
  size = 'medium',
  text = 'Loading...',
}: LoadingSpinnerProps): JSX.Element {
  return (
    <div
      className={`loading-spinner loading-spinner--${size}`}
      role="status"
      aria-live="polite"
      aria-label={text || 'Loading'}
    >
      <div className="loading-spinner__circle" aria-hidden="true" />
      {text && <p className="loading-spinner__text" aria-hidden="true">{text}</p>}
    </div>
  );
}
