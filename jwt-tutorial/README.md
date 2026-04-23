# JWT tutorial

A minimal Node/Express app that shows, end-to-end, how to do JWT auth the
secure way. Everything is intentionally small so you can read the whole
thing in ~15 minutes.

## Run it

```bash
cd jwt-tutorial
npm install
cp .env.example .env
# put a long random string in JWT_SECRET
npm start
# open http://localhost:3001
```

Login with `alice` / `password123` and click the buttons in order. Watch
the Network tab — see when cookies are sent, when the `Authorization`
header is sent, and what the server responds with.

## Map from "the rules" to the code

| Rule | Where it lives |
| --- | --- |
| Refresh token in HttpOnly cookie | `routes/auth.js` → `setRefreshCookie` (`httpOnly: true`) |
| Access token in memory (not storage) | `public/index.html` → `let accessToken = null` |
| HTTPS only | `server.js` → HTTPS redirect + HSTS in production |
| Rotate refresh tokens | `routes/auth.js` → `/auth/refresh` issues a new pair and marks the old `revoked` |
| Refresh tokens in a store with user id, device, expiry | `tokenStore.js` |
| Detect reuse → revoke all sessions | `routes/auth.js` → `revokeAllForUser` on reuse |
| Pinned `algorithms` on verify | `utils/tokens.js` → `verifyAccessToken` passes `algorithms: [config.alg]` |
| RS256 option | `config.js` + `scripts/generateKeys.js` |
| CSRF protection | `middleware/csrf.js` (double-submit cookie) |
| Login rate limit | `routes/auth.js` → `loginLimiter` |

## Why each rule matters (quick reference)

- **HttpOnly cookie for refresh.** JS can't read it, so an XSS payload
  can't exfiltrate it. The cookie rides along only on `/auth/*` requests
  because of `path: '/auth'`.
- **Access token in memory.** If it's in localStorage, any XSS reads it.
  In memory, XSS on the same page can still see it — but it dies on
  reload and can't be read from another origin or tab.
- **Rotation.** If a refresh token leaks, the first use wins. The second
  use (by attacker OR legit user) is proof of theft — the server revokes
  every session for that user.
- **Server-side store.** Lets you revoke individual sessions, list
  devices, or kill everything on password change.
- **Pinned `algorithms`.** Without it, `jsonwebtoken` will accept
  whatever `alg` the token's header claims — including `none` (no sig)
  or HS256 using your RS256 public key as the HMAC secret. Always pass
  `algorithms: ['HS256']` (or `['RS256']`).
- **RS256.** Private key signs, public key verifies. Your auth service
  keeps the private key; every other microservice only needs the public
  key. A compromise of a verifier doesn't let the attacker mint tokens.
- **CSRF.** Cookies are auto-sent on cross-site requests, so a malicious
  page could trigger `/auth/refresh` in the background. The double-submit
  token (cookie value must equal `X-CSRF-Token` header) blocks it
  because cross-origin JS can't read our cookies. `SameSite=Strict` on
  the refresh cookie is the belt; CSRF tokens are the suspenders.
- **Rate limit.** 5 login attempts/minute/IP is enough to make online
  password guessing impractical.

## Switching to RS256

```bash
npm run gen-keys
# edit .env: JWT_ALG=RS256
npm start
```

The private key stays on this service. Give `keys/public.pem` to any
downstream service that needs to verify tokens — they can do so with no
network call back to you.

## What this app intentionally leaves out

- Real database. `USERS` is an array and `tokenStore` is a `Map`.
- Email verification, password reset, MFA.
- Refresh-token binding to IP / user-agent fingerprints (useful but
  breaks mobile networks and enterprise proxies — tune carefully).
- A real HTTPS setup. In production: terminate TLS at your load balancer
  and make sure `trust proxy` matches your hop count.
