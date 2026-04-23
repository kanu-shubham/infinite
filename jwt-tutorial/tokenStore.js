// In a real app this would be Redis or a `refresh_tokens` DB table.
// Keeping it in-memory here so the tutorial runs with zero setup.
//
// Schema per record:
//   jti       -> unique id for this refresh token (also embedded in cookie)
//   userId    -> who it belongs to
//   tokenHash -> sha256 of the opaque refresh token (never store raw)
//   device    -> user-agent / device label, so users can see active sessions
//   expiresAt -> ms epoch
//   revoked   -> bool (set true on logout OR on detected reuse)
//   replacedBy-> jti of the token that superseded this one (for reuse detection)

const store = new Map(); // jti -> record

function save(record) {
  store.set(record.jti, record);
}

function get(jti) {
  const r = store.get(jti);
  if (!r) return null;
  if (r.expiresAt < Date.now()) {
    store.delete(jti);
    return null;
  }
  return r;
}

function revoke(jti) {
  const r = store.get(jti);
  if (r) r.revoked = true;
}

// When theft is detected (an already-rotated refresh token is presented
// again), blow away every session for that user. That's the whole point of
// rotation: the attacker AND the legitimate user both get logged out, and
// the user notices they were kicked out — a signal something is wrong.
function revokeAllForUser(userId) {
  for (const r of store.values()) {
    if (r.userId === userId) r.revoked = true;
  }
}

function listForUser(userId) {
  return [...store.values()].filter((r) => r.userId === userId && !r.revoked);
}

module.exports = { save, get, revoke, revokeAllForUser, listForUser };
