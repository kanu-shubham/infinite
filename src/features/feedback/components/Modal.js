import React, { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import { useFocusTrap } from '../hooks/useFocusTrap';
import { useEscapeKey } from '../hooks/useEscapeKey';
import './Modal.css';

let portalRoot = null;
function getPortalRoot() {
  if (typeof document === 'undefined') return null;
  if (portalRoot && document.body.contains(portalRoot)) return portalRoot;
  portalRoot = document.getElementById('feedback-portal');
  if (!portalRoot) {
    portalRoot = document.createElement('div');
    portalRoot.setAttribute('id', 'feedback-portal');
    document.body.appendChild(portalRoot);
  }
  return portalRoot;
}

/**
 * Accessible modal primitive. Owns:
 *   - portal mount (so stacking-context / overflow:hidden parents don't clip),
 *   - focus trap + restoration,
 *   - ESC + optional backdrop dismiss,
 *   - aria-modal + labelling.
 *
 * Variants are styled by the `variant` class; this primitive doesn't know
 * about feedback semantics.
 */
export function Modal({
  open,
  onClose,
  labelledBy,
  describedBy,
  variant = 'default',
  dismissable = true,
  showClose = true,
  role = 'dialog',
  children,
}) {
  const ref = useRef(null);
  useFocusTrap(ref, open);
  useEscapeKey(() => dismissable && onClose?.(), open);

  // Lock scroll while open. Multiple modals overlap rarely; if they do,
  // we still restore correctly because we capture the prior value.
  useEffect(() => {
    if (!open || typeof document === 'undefined') return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open) return null;
  const root = getPortalRoot();
  if (!root) return null;

  const onBackdropClick = (e) => {
    if (!dismissable) return;
    if (e.target === e.currentTarget) onClose?.();
  };

  return ReactDOM.createPortal(
    <div className="fb-backdrop" onMouseDown={onBackdropClick} data-testid="fb-backdrop">
      <div
        ref={ref}
        role={role}
        aria-modal="true"
        aria-labelledby={labelledBy}
        aria-describedby={describedBy}
        tabIndex={-1}
        className={`fb-modal fb-modal--${variant}`}
        onMouseDown={(e) => e.stopPropagation()}
      >
        {showClose && (
          <button
            type="button"
            className="fb-modal__close"
            aria-label="Close"
            onClick={onClose}
            data-testid="fb-close"
          >
            ×
          </button>
        )}
        {children}
      </div>
    </div>,
    root,
  );
}
