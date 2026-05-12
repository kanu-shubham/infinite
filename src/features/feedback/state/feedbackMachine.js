/**
 * Pure finite-state machine for the feedback flow.
 *
 * Keeping the transitions in a single pure reducer lets us:
 *   - exhaustively unit-test every transition without rendering React,
 *   - reason about which states are reachable (no orphans, no dead ends),
 *   - reuse the same machine if we later swap React for another renderer.
 */

export const STATUS = Object.freeze({
  CLOSED: 'CLOSED',
  RATING: 'RATING',
  NEGATIVE_FORM: 'NEGATIVE_FORM',
  SUBMITTING: 'SUBMITTING',
  THANK_YOU: 'THANK_YOU',
  TRUSTPILOT: 'TRUSTPILOT',
});

export const RATING = Object.freeze({
  NEGATIVE: 'NEGATIVE',
  POSITIVE: 'POSITIVE',
  STELLAR: 'STELLAR',
});

export const initialState = Object.freeze({
  status: STATUS.CLOSED,
  rating: null,
  comment: '',
  error: null,
});

export const ACTION = Object.freeze({
  OPEN: 'OPEN',
  CLOSE: 'CLOSE',
  RATE: 'RATE',
  SUBMIT_START: 'SUBMIT_START',
  SUBMIT_SUCCESS: 'SUBMIT_SUCCESS',
  SUBMIT_ERROR: 'SUBMIT_ERROR',
  THANK_YOU_DONE: 'THANK_YOU_DONE',
});

export function reducer(state, action) {
  switch (action.type) {
    case ACTION.OPEN:
      return { ...initialState, status: STATUS.RATING };

    case ACTION.CLOSE:
      return { ...initialState, status: STATUS.CLOSED };

    case ACTION.RATE: {
      if (state.status !== STATUS.RATING) return state;
      if (action.rating === RATING.NEGATIVE) {
        return { ...state, rating: action.rating, status: STATUS.NEGATIVE_FORM };
      }
      return { ...state, rating: action.rating, status: STATUS.THANK_YOU };
    }

    case ACTION.SUBMIT_START:
      if (state.status !== STATUS.NEGATIVE_FORM) return state;
      return { ...state, status: STATUS.SUBMITTING, error: null };

    case ACTION.SUBMIT_SUCCESS:
      if (state.status !== STATUS.SUBMITTING) return state;
      return { ...state, comment: action.comment ?? '', status: STATUS.THANK_YOU };

    case ACTION.SUBMIT_ERROR:
      if (state.status !== STATUS.SUBMITTING) return state;
      return { ...state, status: STATUS.NEGATIVE_FORM, error: action.error ?? 'Submission failed' };

    case ACTION.THANK_YOU_DONE:
      if (state.status !== STATUS.THANK_YOU) return state;
      return state.rating === RATING.STELLAR
        ? { ...state, status: STATUS.TRUSTPILOT }
        : { ...initialState, status: STATUS.CLOSED };

    default:
      return state;
  }
}
