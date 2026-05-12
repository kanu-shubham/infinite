/**
 * Network boundary for feedback submission.
 *
 * Exposed as a function (not a class) so callers can dependency-inject a
 * mock during tests without monkey-patching modules. In production the
 * widget calls submitFeedback; in tests we pass a fake submit prop.
 */

const MAX_COMMENT_LENGTH = 2000;

export class FeedbackValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FeedbackValidationError';
  }
}

export function sanitizeComment(raw) {
  if (typeof raw !== 'string') return '';
  // Trim and clamp. React already escapes when rendering, so XSS via DOM
  // injection is not a concern here — but server-side, anything goes, so
  // we still bound the payload to avoid abuse.
  return raw.trim().slice(0, MAX_COMMENT_LENGTH);
}

export async function submitFeedback({ rating, comment }, { fetchImpl = fetch, endpoint = '/api/feedback' } = {}) {
  if (!rating) throw new FeedbackValidationError('rating is required');

  const body = JSON.stringify({ rating, comment: sanitizeComment(comment) });

  const res = await fetchImpl(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
  });

  if (!res.ok) {
    throw new Error(`Feedback submission failed: ${res.status}`);
  }
  return res.json().catch(() => ({}));
}

export const __test__ = { MAX_COMMENT_LENGTH };
