"use strict";

// Feature engineering for the demand model. Two key principles:
//   1. The price feature is log(price) so the linear model recovers a
//      constant-elasticity relationship without nonlinear terms.
//   2. Categorical features (location, day-of-week) are one-hot encoded with a
//      stable feature index so online updates and offline retraining share a
//      schema. New unseen categories silently bucket to "other".
//
// FEATURE_NAMES is exported so the policy layer can audit that no protected
// attribute was passed in.

const LOCATIONS = [
  "New York, NY",
  "Los Angeles, CA",
  "Miami, FL",
  "Chicago, IL",
  "San Francisco, CA",
  "Seattle, WA",
  "Austin, TX",
  "Denver, CO",
  "Boston, MA",
  "Nashville, TN",
  "Portland, OR",
  "San Diego, CA",
];

const LOCATION_INDEX = Object.fromEntries(LOCATIONS.map((l, i) => [l, i]));

function dayOfWeek(dateIso) {
  const d = new Date(dateIso);
  return Number.isFinite(d.getTime()) ? d.getUTCDay() : 0;
}

function buildContextFeatures({ hotel, context = {} }) {
  const locOH = Array(LOCATIONS.length).fill(0);
  const idx = LOCATION_INDEX[hotel.location];
  if (idx != null) locOH[idx] = 1;

  const dow = dayOfWeek(context.checkIn || context.date);
  const dowOH = Array(7).fill(0);
  dowOH[dow] = 1;

  const isWeekend = dow === 0 || dow === 6 ? 1 : 0;
  const lengthOfStay = Math.max(1, Number(context.nights) || 1);
  const leadTimeDays = Math.max(0, Number(context.leadTimeDays) || 0);

  return [
    1, // bias / intercept
    hotel.rating / 5,
    Math.log1p(hotel.reviewCount),
    (hotel.amenities ? hotel.amenities.length : 0) / 12,
    isWeekend,
    Math.log1p(lengthOfStay),
    Math.log1p(leadTimeDays),
    ...locOH,
    ...dowOH,
  ];
}

const FEATURE_NAMES = [
  "bias",
  "rating_norm",
  "log1p_reviews",
  "amenity_density",
  "is_weekend",
  "log1p_nights",
  "log1p_lead_time",
  ...LOCATIONS.map((l) => `loc_${l.replace(/[^a-z0-9]/gi, "_").toLowerCase()}`),
  ...["sun", "mon", "tue", "wed", "thu", "fri", "sat"].map((d) => `dow_${d}`),
];

const CONTEXT_DIM = FEATURE_NAMES.length;

// Full feature for demand model = [contextFeatures..., logPrice]. Keeping
// logPrice as the LAST coordinate makes elasticity == weights[CONTEXT_DIM].
function buildDemandFeatures({ hotel, context, price }) {
  const ctx = buildContextFeatures({ hotel, context });
  return [...ctx, Math.log(Math.max(price, 1e-6))];
}

const DEMAND_FEATURE_NAMES = [...FEATURE_NAMES, "log_price"];
const DEMAND_DIM = DEMAND_FEATURE_NAMES.length;

module.exports = {
  buildContextFeatures,
  buildDemandFeatures,
  FEATURE_NAMES,
  DEMAND_FEATURE_NAMES,
  CONTEXT_DIM,
  DEMAND_DIM,
  LOCATIONS,
};
