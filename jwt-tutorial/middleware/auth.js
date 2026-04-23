const { verifyAccessToken } = require('../utils/tokens');

// Pulls the access token out of the `Authorization: Bearer <token>` header.
// We deliberately do NOT read the access token from a cookie — the client
// holds it in memory and attaches it to each request. That means an
// attacker who steals the refresh cookie via CSRF still can't call
// protected APIs (the cookie is only accepted by /auth/refresh).
function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const [scheme, token] = header.split(' ');
  if (scheme !== 'Bearer' || !token) {
    return res.status(401).json({ error: 'missing bearer token' });
  }
  try {
    req.user = verifyAccessToken(token); // pinned-algorithm verify
    next();
  } catch (err) {
    return res.status(401).json({ error: 'invalid or expired token' });
  }
}

module.exports = { requireAuth };
