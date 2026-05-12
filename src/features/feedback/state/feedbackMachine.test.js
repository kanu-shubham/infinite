import {
  ACTION,
  RATING,
  STATUS,
  initialState,
  reducer,
} from './feedbackMachine';

describe('feedbackMachine', () => {
  test('OPEN moves from CLOSED → RATING', () => {
    const next = reducer(initialState, { type: ACTION.OPEN });
    expect(next.status).toBe(STATUS.RATING);
    expect(next.rating).toBeNull();
  });

  test('CLOSE resets state', () => {
    const open = reducer(initialState, { type: ACTION.OPEN });
    const rated = reducer(open, { type: ACTION.RATE, rating: RATING.STELLAR });
    const closed = reducer(rated, { type: ACTION.CLOSE });
    expect(closed).toEqual({ ...initialState, status: STATUS.CLOSED });
  });

  describe('RATE', () => {
    test('NEGATIVE → NEGATIVE_FORM', () => {
      const open = reducer(initialState, { type: ACTION.OPEN });
      const next = reducer(open, { type: ACTION.RATE, rating: RATING.NEGATIVE });
      expect(next.status).toBe(STATUS.NEGATIVE_FORM);
      expect(next.rating).toBe(RATING.NEGATIVE);
    });

    test.each([RATING.POSITIVE, RATING.STELLAR])('%s → THANK_YOU', (rating) => {
      const open = reducer(initialState, { type: ACTION.OPEN });
      const next = reducer(open, { type: ACTION.RATE, rating });
      expect(next.status).toBe(STATUS.THANK_YOU);
      expect(next.rating).toBe(rating);
    });

    test('ignored when not in RATING state', () => {
      const open = reducer(initialState, { type: ACTION.OPEN });
      const rated = reducer(open, { type: ACTION.RATE, rating: RATING.NEGATIVE });
      const again = reducer(rated, { type: ACTION.RATE, rating: RATING.POSITIVE });
      expect(again).toEqual(rated);
    });
  });

  describe('submit flow', () => {
    const inForm = () => {
      const o = reducer(initialState, { type: ACTION.OPEN });
      return reducer(o, { type: ACTION.RATE, rating: RATING.NEGATIVE });
    };

    test('SUBMIT_START → SUBMITTING (clears error)', () => {
      const start = { ...inForm(), error: 'prev' };
      const next = reducer(start, { type: ACTION.SUBMIT_START });
      expect(next.status).toBe(STATUS.SUBMITTING);
      expect(next.error).toBeNull();
    });

    test('SUBMIT_SUCCESS → THANK_YOU with comment', () => {
      const submitting = reducer(inForm(), { type: ACTION.SUBMIT_START });
      const next = reducer(submitting, { type: ACTION.SUBMIT_SUCCESS, comment: 'broken' });
      expect(next.status).toBe(STATUS.THANK_YOU);
      expect(next.comment).toBe('broken');
    });

    test('SUBMIT_ERROR → NEGATIVE_FORM with error', () => {
      const submitting = reducer(inForm(), { type: ACTION.SUBMIT_START });
      const next = reducer(submitting, { type: ACTION.SUBMIT_ERROR, error: 'oops' });
      expect(next.status).toBe(STATUS.NEGATIVE_FORM);
      expect(next.error).toBe('oops');
    });

    test('SUBMIT_START ignored when not in NEGATIVE_FORM', () => {
      const next = reducer(initialState, { type: ACTION.SUBMIT_START });
      expect(next).toBe(initialState);
    });
  });

  describe('THANK_YOU_DONE', () => {
    test('STELLAR → TRUSTPILOT', () => {
      const o = reducer(initialState, { type: ACTION.OPEN });
      const stellar = reducer(o, { type: ACTION.RATE, rating: RATING.STELLAR });
      const next = reducer(stellar, { type: ACTION.THANK_YOU_DONE });
      expect(next.status).toBe(STATUS.TRUSTPILOT);
    });

    test('POSITIVE → CLOSED', () => {
      const o = reducer(initialState, { type: ACTION.OPEN });
      const positive = reducer(o, { type: ACTION.RATE, rating: RATING.POSITIVE });
      const next = reducer(positive, { type: ACTION.THANK_YOU_DONE });
      expect(next.status).toBe(STATUS.CLOSED);
    });

    test('NEGATIVE after submit → CLOSED', () => {
      const o = reducer(initialState, { type: ACTION.OPEN });
      const neg = reducer(o, { type: ACTION.RATE, rating: RATING.NEGATIVE });
      const submitting = reducer(neg, { type: ACTION.SUBMIT_START });
      const thanks = reducer(submitting, { type: ACTION.SUBMIT_SUCCESS, comment: 'hi' });
      const next = reducer(thanks, { type: ACTION.THANK_YOU_DONE });
      expect(next.status).toBe(STATUS.CLOSED);
    });
  });

  test('unknown action returns same reference', () => {
    expect(reducer(initialState, { type: 'WAT' })).toBe(initialState);
  });
});
