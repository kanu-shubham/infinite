// Generates an RSA keypair for RS256 signing.
// Run: npm run gen-keys  (then set JWT_ALG=RS256 in .env)
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const dir = path.join(__dirname, '..', 'keys');
fs.mkdirSync(dir, { recursive: true });

const { privateKey, publicKey } = crypto.generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

fs.writeFileSync(path.join(dir, 'private.pem'), privateKey, { mode: 0o600 });
fs.writeFileSync(path.join(dir, 'public.pem'), publicKey);

console.log('Wrote keys/private.pem (0600) and keys/public.pem');
console.log('Give the public key to any service that needs to verify tokens.');
console.log('The private key stays on the signing service only.');
