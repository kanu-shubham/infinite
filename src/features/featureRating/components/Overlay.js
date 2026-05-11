import React from "react";
import "./Overlay.css";

export default function Overlay({ onClick, children }) {
  return (
    <div className="overlay" role="dialog" aria-modal="true">
      <div className="overlay__backdrop" onClick={onClick} aria-hidden="true" />
      <div className="overlay__content">{children}</div>
    </div>
  );
}
