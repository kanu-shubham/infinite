"use strict";

const { makeRng, gaussian } = require("../ml/random");
const { buildContextFeatures, LOCATIONS } = require("../ml/features");

// Generates a small synthetic hotel catalog and a corpus of historical
// (price, units, context) tuples with a KNOWN ground-truth elasticity per
// location. Tests use this to verify the elasticity estimator recovers the
// true value from biased logs.

const HOTEL_NAMES = [
  "The Grand Palace",
  "Ocean View Resort",
  "Mountain Lodge",
  "City Center Inn",
  "Sunset Beach Hotel",
  "Royal Garden Suites",
  "Harbor Light Inn",
  "Pine Valley Resort",
  "The Sapphire Hotel",
  "Riverside Retreat",
  "Cloud Nine Suites",
  "Golden Sands Resort",
];

function seedHotels(repo, count = 12) {
  const rng = makeRng(7);
  const amenitiesAll = [
    "Free WiFi",
    "Pool",
    "Spa",
    "Gym",
    "Restaurant",
    "Bar",
    "Room Service",
    "Parking",
    "Pet Friendly",
    "Beach Access",
    "Business Center",
    "Concierge",
  ];
  for (let i = 0; i < count; i++) {
    const id = i + 1;
    const rating = Math.round((3 + rng() * 2) * 10) / 10;
    const reviewCount = Math.round(50 + rng() * 950);
    const basePrice = Math.round(80 + rng() * 320);
    const amenityCount = 3 + Math.floor(rng() * 6);
    const shuffled = [...amenitiesAll].sort(() => rng() - 0.5);
    repo.upsertHotel({
      id,
      name: HOTEL_NAMES[i % HOTEL_NAMES.length],
      location: LOCATIONS[i % LOCATIONS.length],
      basePrice,
      rating,
      reviewCount,
      amenities: shuffled.slice(0, amenityCount),
    });
  }
}

// Ground-truth demand: log(units) = a(hotel) + e(loc) * log(price) + ctx*gamma + noise
// Historical price logging policy: noisy uniform around base price (so we get
// some price variation — without it elasticity is unidentifiable).
function generateHistory(repo, samplesPerHotel = 60, seed = 13) {
  const rng = makeRng(seed);
  const hotels = repo.listHotels();
  const truth = new Map();

  for (const h of hotels) {
    const elasticity = -1.4 - rng() * 1.0; // [-2.4, -1.4]
    // Intercept on log(units): chosen so that at p ~ basePrice expected units
    // is in the ~3-15 range for a clean signal (avoids rounding to zero).
    const logBase = 9 + rng() * 4;
    truth.set(h.id, { elasticity, logBase });
  }

  const logs = [];
  const now = Date.now();
  for (const h of hotels) {
    const t = truth.get(h.id);
    for (let i = 0; i < samplesPerHotel; i++) {
      const daysBack = Math.floor(rng() * 90);
      const checkIn = new Date(now - daysBack * 86400000).toISOString();
      const nights = 1 + Math.floor(rng() * 4);
      const leadTimeDays = Math.floor(rng() * 60);
      const ctx = { checkIn, nights, leadTimeDays };

      // Logging policy: bell curve around base price, +/- 35%.
      const noise = 0.35 * gaussian(rng);
      const price = Math.max(20, Math.round(h.basePrice * (1 + noise)));
      const propensity = 1 / (h.basePrice * 0.7); // crude uniform-density proxy

      const ctxFeatures = buildContextFeatures({ hotel: h, context: ctx });
      const ctxContribution =
        0.05 * ctxFeatures[1] + // rating bump
        0.02 * ctxFeatures[4] + // weekend bump
        0.01 * ctxFeatures[5]; // length-of-stay
      const logUnits =
        t.logBase +
        t.elasticity * Math.log(price) +
        ctxContribution +
        0.15 * gaussian(rng);
      const units = Math.max(0, Math.round(Math.exp(logUnits)));
      const booked = units > 0 ? 1 : 0;

      logs.push({
        hotelId: h.id,
        price,
        units,
        booked,
        propensity,
        context: ctx,
        decisionId: `seed-${h.id}-${i}`,
        ts: now - daysBack * 86400000,
      });
    }
  }

  for (const log of logs) repo.recordFeedback(log);
  return { truth, logs };
}

module.exports = { seedHotels, generateHistory };
