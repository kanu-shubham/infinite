const express = require('express');
const bcrypt = require('bcryptjs');
const rateLimit = require('express-rate-limit');

const config = require('../config');
const tokenStore = require('../tokenStore');
const { issueCsrfCookie, requireCsrf } = require('../middleware/csrf');
const {
  signAccessToken,
  generateRefreshToken,
  hashRefreshToken,
} = require('../utils/tokens');

const router = express.Router();

// --- fake user DB. In reality: query Postgres, etc. ---
const USERS = [
  {
    id: 'u1',
    username: 'alice',
    // password is "password123" — bcrypt-hashed at startup for demo
    passwordHash: bcrypt.hashSync('password123', 10),
  },
];

// --- brute-force defense: cap login attempts per IP ---
// Tight window/limit for a teaching demo; real apps tune per risk profile
// and usually combine IP + username buckets.
const loginLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 5,              // 5 attempts per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'too many login attempts, slow down' },
});

const REFRESH_COOKIE = 'refreshToken';

function setRefreshCookie(res, rawToken, jti) {
  // HttpOnly      => JS on the page cannot read it (XSS can't steal it).
  // Secure        => only sent over HTTPS in prod.
  // SameSite=Strict => never sent on cross-site requests at all; the
  //                    strongest CSRF defense for this particular cookie.
  //                    (We still add CSRF tokens for defense in depth.)
  // Path          => cookie is ONLY sent to /auth endpoints, so random
  //                    API calls don't leak it into server logs.
  res.cookie(REFRESH_COOKIE, `${jti}.${rawToken}`, {
    httpOnly: true,
    secure: config.isProd,
    sameSite: 'strict',
    path: '/auth',
    maxAge: config.refreshTtlDays * 24 * 60 * 60 * 1000,
  });
}

function clearRefreshCookie(res) {
  res.clearCookie(REFRESH_COOKIE, { path: '/auth' });
}

async function issueSession(res, user, device) {
  const raw = generateRefreshToken();
  const jti = require('crypto').randomBytes(12).toString('hex');

  tokenStore.save({
    jti,
    userId: user.id,
    tokenHash: hashRefreshToken(raw),
    device,
    expiresAt: Date.now() + config.refreshTtlDays * 24 * 60 * 60 * 1000,
    revoked: false,
    replacedBy: null,
  });

  setRefreshCookie(res, raw, jti);
  issueCsrfCookie(res);

  const accessToken = signAccessToken({ sub: user.id, username: user.username });
  return { accessToken };
}

// ---------------- POST /auth/login ----------------
router.post('/login', loginLimiter, async (req, res) => {
  const { username, password } = req.body || {};
  const user = USERS.find((u) => u.username === username);

  // Always run bcrypt even if user is missing, to keep response time roughly
  // constant and not leak which usernames exist.
  const ok = user
    ? await bcrypt.compare(password || '', user.passwordHash)
    : await bcrypt.compare('dummy', '$2a$10$CwTycUXWue0Thq9StjUM0uJ8.X.5uQm1Z5Oj0V2U0eZ4p2hHq2YyC');

  if (!user || !ok) {
    return res.status(401).json({ error: 'invalid credentials' });
  }

  const { accessToken } = await issueSession(res, user, req.headers['user-agent'] || 'unknown');
  res.json({ accessToken, user: { id: user.id, username: user.username } });
});

// ---------------- POST /auth/refresh ----------------
// Rotates the refresh token: every refresh invalidates the old one and
// issues a new pair. If the OLD one is ever presented again, we assume
// it was stolen and nuke every session for that user.
router.post('/refresh', requireCsrf, async (req, res) => {
  const cookie = req.cookies?.[REFRESH_COOKIE];
  if (!cookie) return res.status(401).json({ error: 'no refresh token' });

  const [jti, raw] = cookie.split('.');
  const record = tokenStore.get(jti);

  if (!record) {
    clearRefreshCookie(res);
    return res.status(401).json({ error: 'unknown refresh token' });
  }

  // Reuse detection: this jti was already rotated away.
  if (record.revoked || record.replacedBy) {
    tokenStore.revokeAllForUser(record.userId);
    clearRefreshCookie(res);
    return res.status(401).json({ error: 'refresh token reuse detected; all sessions revoked' });
  }

  if (hashRefreshToken(raw) !== record.tokenHash) {
    clearRefreshCookie(res);
    return res.status(401).json({ error: 'bad refresh token' });
  }

  const user = USERS.find((u) => u.id === record.userId);
  if (!user) return res.status(401).json({ error: 'user gone' });

  // Rotate: mark old as replaced, issue new.
  const { accessToken } = await issueSession(res, user, record.device);
  record.revoked = true;
  record.replacedBy = 'rotated';

  res.json({ accessToken, user: { id: user.id, username: user.username } });
});

// ---------------- POST /auth/logout ----------------
router.post('/logout', requireCsrf, (req, res) => {
  const cookie = req.cookies?.[REFRESH_COOKIE];
  if (cookie) {
    const [jti] = cookie.split('.');
    tokenStore.revoke(jti);
  }
  clearRefreshCookie(res);
  res.json({ ok: true });
});

module.exports = router;
