import React from 'react';
import './ErrorMessage.css';

export interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorMessage({ message, onRetry }: ErrorMessageProps): JSX.Element {
  return (
    <div className="error-message">
      <div className="error-message__icon">!</div>
      <p className="error-message__text">{message}</p>
      {onRetry && (
        <button className="error-message__retry" onClick={onRetry}>
          Try Again
        </button>
      )}
    </div>
  );
}
