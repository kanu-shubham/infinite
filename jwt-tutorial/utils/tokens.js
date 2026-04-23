const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const config = require('../config');

// Access tokens are SHORT-LIVED JWTs. They go to the client in the JSON
// response body and are held in memory (never localStorage). If the tab
// closes they vanish — that's fine, the refresh cookie mints a new one.
function signAccessToken(payload) {
  return jwt.sign(payload, config.signingKey, {
    algorithm: config.alg,
    expiresIn: config.accessTtl,
  });
}

// Verify with a PINNED algorithm list. Without this, an attacker could
// present a token with `alg: none` (no signature) or downgrade an RS256
// token to HS256 using the public key as the HMAC secret. Both are classic
// jsonwebtoken footguns — always pass `algorithms`.
function verifyAccessToken(token) {
  return jwt.verify(token, config.verificationKey, {
    algorithms: [config.alg],
  });
}

// Refresh tokens are NOT JWTs here. They are opaque random strings whose
// hash is stored server-side. That way:
//   - the cookie value is useless without the DB record
//   - we can revoke individual sessions (delete the row)
//   - stolen-token reuse can be detected (see routes/auth.js refresh)
function generateRefreshToken() {
  return crypto.randomBytes(48).toString('hex');
}

function hashRefreshToken(token) {
  return crypto.createHash('sha256').update(token).digest('hex');
}

module.exports = {
  signAccessToken,
  verifyAccessToken,
  generateRefreshToken,
  hashRefreshToken,
};
