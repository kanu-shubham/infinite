const express = require('express');
const { requireAuth } = require('../middleware/auth');
const tokenStore = require('../tokenStore');

const router = express.Router();

router.get('/me', requireAuth, (req, res) => {
  res.json({ user: req.user });
});

// Returns the list of refresh tokens (sessions) this user currently has,
// so they can see every device that's logged in — a feature users love
// and one you get for free once you store refresh tokens server-side.
router.get('/sessions', requireAuth, (req, res) => {
  const sessions = tokenStore.listForUser(req.user.sub).map((r) => ({
    jti: r.jti,
    device: r.device,
    expiresAt: new Date(r.expiresAt).toISOString(),
  }));
  res.json({ sessions });
});

module.exports = router;
