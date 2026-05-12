import { submitFeedback, sanitizeComment, FeedbackValidationError } from './feedbackService';

describe('sanitizeComment', () => {
  test('trims whitespace', () => {
    expect(sanitizeComment('   hello  ')).toBe('hello');
  });

  test('caps length at 2000 chars', () => {
    expect(sanitizeComment('a'.repeat(5000)).length).toBe(2000);
  });

  test('coerces non-strings to empty', () => {
    expect(sanitizeComment(null)).toBe('');
    expect(sanitizeComment(undefined)).toBe('');
    expect(sanitizeComment(42)).toBe('');
  });
});

describe('submitFeedback', () => {
  test('throws when rating missing', async () => {
    await expect(submitFeedback({})).rejects.toBeInstanceOf(FeedbackValidationError);
  });

  test('POSTs JSON to endpoint and returns response body', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: 'abc' }),
    });
    const out = await submitFeedback(
      { rating: 'NEGATIVE', comment: '  bad  ' },
      { fetchImpl, endpoint: '/x' },
    );
    expect(fetchImpl).toHaveBeenCalledWith('/x', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating: 'NEGATIVE', comment: 'bad' }),
    }));
    expect(out).toEqual({ id: 'abc' });
  });

  test('throws on non-OK response', async () => {
    const fetchImpl = jest.fn().mockResolvedValue({ ok: false, status: 500, json: () => ({}) });
    await expect(submitFeedback({ rating: 'POSITIVE' }, { fetchImpl })).rejects.toThrow(/500/);
  });
});
