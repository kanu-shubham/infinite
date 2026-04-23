require('dotenv').config();
const fs = require('fs');
const path = require('path');

const alg = process.env.JWT_ALG || 'HS256';
const isProd = process.env.NODE_ENV === 'production';

// HS256 = symmetric: same secret signs and verifies. Simple, but any service
// that can verify can also forge. Good for a single server.
// RS256 = asymmetric: private key signs, public key verifies. Multiple
// services can verify without ever seeing the signing key.
let signingKey;
let verificationKey;

if (alg === 'RS256') {
  const privPath = path.join(__dirname, 'keys', 'private.pem');
  const pubPath = path.join(__dirname, 'keys', 'public.pem');
  if (!fs.existsSync(privPath) || !fs.existsSync(pubPath)) {
    throw new Error('RS256 selected but keys/ missing. Run: npm run gen-keys');
  }
  signingKey = fs.readFileSync(privPath);
  verificationKey = fs.readFileSync(pubPath);
} else {
  if (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32) {
    throw new Error('JWT_SECRET must be set and at least 32 chars.');
  }
  signingKey = process.env.JWT_SECRET;
  verificationKey = process.env.JWT_SECRET;
}

module.exports = {
  alg,
  isProd,
  port: Number(process.env.PORT) || 3001,
  signingKey,
  verificationKey,
  accessTtl: process.env.ACCESS_TOKEN_TTL || '15m',
  refreshTtlDays: Number(process.env.REFRESH_TOKEN_TTL_DAYS) || 7,
};
