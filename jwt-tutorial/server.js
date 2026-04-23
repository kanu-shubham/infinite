const express = require('express');
const cookieParser = require('cookie-parser');
const path = require('path');

const config = require('./config');
const authRoutes = require('./routes/auth');
const protectedRoutes = require('./routes/protected');

const app = express();

// Behind a reverse proxy (Heroku, nginx, etc.) Express needs this so that
// `req.secure` and rate-limit see the real client IP, not the proxy's.
app.set('trust proxy', 1);

app.use(express.json());
app.use(cookieParser());

// HTTPS-only redirect in production. Tokens in transit over plain HTTP can
// be sniffed on any hop — hotel wifi, a misconfigured proxy, etc. In dev
// we skip this so http://localhost works.
app.use((req, res, next) => {
  if (config.isProd && req.headers['x-forwarded-proto'] !== 'https') {
    return res.redirect(301, `https://${req.headers.host}${req.url}`);
  }
  // HSTS tells browsers "only ever talk to me over HTTPS" for 6 months.
  if (config.isProd) {
    res.setHeader('Strict-Transport-Security', 'max-age=15552000; includeSubDomains');
  }
  next();
});

// Static demo client
app.use(express.static(path.join(__dirname, 'public')));

app.use('/auth', authRoutes);
app.use('/api', protectedRoutes);

app.listen(config.port, () => {
  console.log(`jwt-tutorial listening on http://localhost:${config.port} (alg=${config.alg})`);
});
