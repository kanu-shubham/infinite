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
    <div className={`loading-spinner loading-spinner--${size}`}>
      <div className="loading-spinner__circle" />
      {text && <p className="loading-spinner__text">{text}</p>}
    </div>
  );
}
