const crypto = require('crypto');
const config = require('../config');

// Double-submit cookie pattern.
//
// Why we need CSRF protection at all: we put the refresh token in a cookie,
// and browsers auto-attach cookies to cross-site requests. Without a second
// check, evil.com could POST to our /auth/refresh and the browser would
// happily send our cookie along.
//
// How double-submit defends us:
//   1. Server sets a `csrfToken` cookie that is readable by JS (NOT HttpOnly).
//   2. Client JS reads that cookie and echoes it back in `X-CSRF-Token`.
//   3. Server requires cookie value === header value.
// Cross-site attacker code can't read our cookie (same-origin policy), so
// it can't forge the matching header.

const CSRF_COOKIE = 'csrfToken';
const CSRF_HEADER = 'x-csrf-token';

function issueCsrfCookie(res) {
  const token = crypto.randomBytes(24).toString('hex');
  res.cookie(CSRF_COOKIE, token, {
    httpOnly: false, // JS must be able to read it
    secure: config.isProd,
    sameSite: 'lax',
    path: '/',
  });
  return token;
}

function requireCsrf(req, res, next) {
  const cookieValue = req.cookies?.[CSRF_COOKIE];
  const headerValue = req.headers[CSRF_HEADER];
  if (!cookieValue || !headerValue || cookieValue !== headerValue) {
    return res.status(403).json({ error: 'csrf check failed' });
  }
  next();
}

module.exports = { issueCsrfCookie, requireCsrf, CSRF_COOKIE, CSRF_HEADER };
