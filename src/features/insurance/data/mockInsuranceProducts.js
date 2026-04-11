/**
 * Mock travel insurance product catalogue.
 * Each product carries a pre-computed 128-dim embedding that encodes its
 * type, coverage, price tier, target traveler and destination profile.
 * In production these embeddings would come from a trained item tower model.
 */

// ─── Lookup tables ────────────────────────────────────────────────────────────

export const PRODUCT_TYPES = [
  "basic",
  "comprehensive",
  "adventure",
  "medical",
  "cfar",
  "family",
  "senior",
  "student",
];

export const TYPE_LABELS = {
  basic:         "Basic",
  comprehensive: "Comprehensive",
  adventure:     "Adventure",
  medical:       "Medical Only",
  cfar:          "Cancel Anytime",
  family:        "Family",
  senior:        "Senior",
  student:       "Student",
};

export const TYPE_COLORS = {
  basic:         "#6366f1",
  comprehensive: "#0ea5e9",
  adventure:     "#f97316",
  medical:       "#ef4444",
  cfar:          "#8b5cf6",
  family:        "#22c55e",
  senior:        "#eab308",
  student:       "#14b8a6",
};

const PROVIDERS = [
  { name: "TravelGuard",     tier: 4, claimScore: 0.88 },
  { name: "Allianz Travel",  tier: 5, claimScore: 0.92 },
  { name: "World Nomads",    tier: 4, claimScore: 0.87 },
  { name: "AXA Assistance",  tier: 4, claimScore: 0.85 },
  { name: "Seven Corners",   tier: 3, claimScore: 0.82 },
  { name: "Generali Global", tier: 3, claimScore: 0.80 },
  { name: "IMG Global",      tier: 4, claimScore: 0.86 },
  { name: "SafeTrip",        tier: 2, claimScore: 0.78 },
];

const NAMES_BY_TYPE = {
  basic:         ["Basic Traveler", "EssentialCover", "TripSafe Lite",    "Journey Basic",    "SimpleProtect",   "QuickShield",    "TravelEase",    "DayTrip Basic",  "EcoTravel",     "BudgetSafe",     "PathFinder Lite", "WanderBasic"],
  comprehensive: ["AllRisk Pro",    "TotalCover Plus","Premier Guard",     "Elite Shield",     "Complete Journey","FullProtect Pro", "PremiumTravel", "Apex Coverage",  "PlatinumPlan",  "GlobalElite",    "MasterShield",    "OmniTravel"],
  adventure:     ["AdventureElite","ExtremeSports Pro","Summit Shield",   "WildTrack Cover",  "Thrill Guard",    "ActiveProtect",  "AdrenalineX",   "VerticalEdge",   "TerrainPro",    "ExpeditionShield","WildernessPlus",  "OutdoorElite"],
  medical:       ["MediTravel Pro","Emergency Plus",  "HealthGuard Global","MedEvac Shield",  "CriticalCare",    "Medical Prime",  "ClinicGuard",   "HospitalPlus",   "MedFirst Pro",  "HealSafe",       "GlobalMedic",     "UrgentCare Pro"],
  cfar:          ["Cancel Anytime","FlexCancel Elite","AnyReason Shield", "FreedomCancel",    "TotalFlex Cover", "OpenCancel Plus","FlexTrip Pro",  "PeaceOfMind",    "NoReasonNeeded","FlexBack Shield","CancelAll Pro",   "RefundReady"],
  family:        ["FamilyFirst",   "KidsIncluded Pro","FamilyJourney Plus","AllFamily Guard", "ParentCover Elite","TripFamily Pro","HomeAway Pro",  "FamilyCircle",   "TravelTribe",   "KidsFirst Shield","BroodProtect",   "FamilyElite"],
  senior:        ["SeniorCare Pro","GoldenTrip Shield","ElderGuard Plus", "MatureTravel Pro", "SeniorFlex Cover","GoldYears",      "PlatinumAge",   "RetiredElite",   "WiseTraveler",  "LifeJourney Pro","EternalExplorer", "SilverShield"],
  student:       ["BackpackerBasic","StudentExplorer","BudgetWanderer",   "YouthTravel Pro",  "GapYear Shield",  "CampusTraveler","StudentElite",  "BackpackPro",    "YouthRoamer",   "DormToWorld",    "FreeSpiritPlan",  "MillennialTrail"],
};

const FEATURES_BY_TYPE = {
  basic:         ["Trip Cancellation", "Baggage Loss", "Travel Delay", "24/7 Assistance"],
  comprehensive: ["Trip Cancellation", "Medical Emergency", "Evacuation", "Baggage Loss", "Travel Delay", "Rental Car", "Cancel For Any Reason", "Pre-existing Conditions"],
  adventure:     ["Medical Emergency", "Evacuation", "Adventure Sports", "Trip Cancellation", "Equipment Loss", "Search & Rescue", "High Altitude Coverage"],
  medical:       ["Medical Emergency", "Emergency Evacuation", "Medical Repatriation", "24/7 Medical Hotline", "Pre-existing Conditions"],
  cfar:          ["Cancel For Any Reason", "Trip Cancellation", "Trip Interruption", "Travel Delay", "Baggage Loss", "Airline Failure"],
  family:        ["Trip Cancellation", "Child Medical", "Family Evacuation", "Baggage Loss", "Travel Delay", "Pet Care Coverage"],
  senior:        ["Medical Emergency", "Pre-existing Conditions", "Evacuation", "Trip Cancellation", "Medical Repatriation", "24/7 Medical Support"],
  student:       ["Trip Cancellation", "Medical Emergency", "Baggage Loss", "Travel Delay", "Electronic Equipment"],
};

const DESTINATIONS_BY_TYPE = {
  basic:         ["domestic", "international"],
  comprehensive: ["domestic", "international", "schengen", "asia", "latam", "africa", "oceania"],
  adventure:     ["international", "adventure", "polar", "africa", "asia"],
  medical:       ["international", "schengen", "asia", "latam", "africa"],
  cfar:          ["domestic", "international", "schengen", "cruise"],
  family:        ["domestic", "international", "cruise", "asia"],
  senior:        ["domestic", "international", "cruise", "schengen"],
  student:       ["international", "asia", "latam", "backpacker"],
};

const STYLES_BY_TYPE = {
  basic:         ["solo", "couple", "short-trip"],
  comprehensive: ["business", "couple", "leisure", "group"],
  adventure:     ["solo", "backpacker", "group", "extreme"],
  medical:       ["senior", "business", "leisure"],
  cfar:          ["business", "couple", "leisure", "group"],
  family:        ["family", "leisure", "cruise"],
  senior:        ["senior", "cruise", "leisure"],
  student:       ["student", "backpacker", "solo"],
};

// Price per day (USD) ranges per type
const PRICE_RANGE = {
  basic:         [1.5,  4.0],
  comprehensive: [5.0, 15.0],
  adventure:     [6.0, 18.0],
  medical:       [3.0,  9.0],
  cfar:          [8.0, 20.0],
  family:        [4.0, 12.0],
  senior:        [6.0, 16.0],
  student:       [1.0,  3.5],
};

const COVERAGE_RANGE = {
  basic:         [15_000,  50_000],
  comprehensive: [50_000, 500_000],
  adventure:     [50_000, 300_000],
  medical:       [100_000, 1_000_000],
  cfar:          [20_000, 150_000],
  family:        [30_000, 200_000],
  senior:        [50_000, 500_000],
  student:       [10_000,  50_000],
};

const BADGES = [null, null, null, "Best Value", null, null, "Editor's Choice", null, null, null, "Most Popular", null];

// ─── Embedding helpers ────────────────────────────────────────────────────────

/**
 * Deterministic seeded PRNG (Mulberry32).
 */
function seededRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalize(vec) {
  const mag = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / mag);
}

/**
 * Encode a product into 128-dim semantic embedding.
 *
 * Layout (128 dims):
 *  [0-7]   product type one-hot
 *  [8-12]  price tier soft-encoding (0=budget … 4=luxury)
 *  [13-17] trip-length compatibility
 *  [18-25] destination coverage flags
 *  [26-41] feature flags (16 features)
 *  [42-49] traveler style flags
 *  [50-55] provider quality signals
 *  [56-63] coverage-amount bins
 *  [64-127] deterministic noise (makes similar products distinguishable)
 */
function computeProductEmbedding(product, seed) {
  const r    = seededRandom(seed);
  const vec  = new Array(128).fill(0);
  const TYPE_IDX = { basic:0, comprehensive:1, adventure:2, medical:3, cfar:4, family:5, senior:6, student:7 };

  // [0-7] type one-hot
  if (TYPE_IDX[product.type] !== undefined) vec[TYPE_IDX[product.type]] = 1.0;

  // [8-12] price tier
  const ppt    = product.pricePerDay;
  const tier   = ppt < 3 ? 0 : ppt < 6 ? 1 : ppt < 10 ? 2 : ppt < 15 ? 3 : 4;
  vec[8 + tier] = 1.0;
  // soft bleed to adjacent tiers
  if (tier > 0) vec[8 + tier - 1] = 0.3;
  if (tier < 4) vec[8 + tier + 1] = 0.3;

  // [13-17] trip length (maxTripDays bins)
  const mtd = product.maxTripDays;
  if (mtd <=  7) vec[13] = 1.0;
  if (mtd <= 14) vec[14] = 1.0;
  if (mtd <= 30) vec[15] = 1.0;
  if (mtd <= 60) vec[16] = 1.0;
  vec[17] = 1.0; // always supports some length

  // [18-25] destination flags
  const DEST_IDX = { domestic:18, international:19, schengen:20, asia:21, latam:22, africa:23, oceania:24, adventure:25, polar:25, cruise:24, backpacker:23 };
  for (const d of product.destinations) if (DEST_IDX[d] !== undefined) vec[DEST_IDX[d]] = 1.0;

  // [26-41] feature flags
  const FEAT_IDX = {
    "Trip Cancellation": 26, "Medical Emergency": 27, "Evacuation": 28, "Baggage Loss": 29,
    "Travel Delay": 30, "Adventure Sports": 31, "Pre-existing Conditions": 32, "Cancel For Any Reason": 33,
    "Rental Car": 34, "Equipment Loss": 35, "Search & Rescue": 36, "Medical Repatriation": 37,
    "24/7 Assistance": 38, "24/7 Medical Hotline": 38, "24/7 Medical Support": 38,
    "Child Medical": 39, "Family Evacuation": 28, "Pet Care Coverage": 40,
    "Electronic Equipment": 41, "High Altitude Coverage": 36, "Airline Failure": 33,
    "Trip Interruption": 26, "Emergency Evacuation": 28,
  };
  for (const f of product.features) if (FEAT_IDX[f] !== undefined) vec[FEAT_IDX[f]] = 1.0;

  // [42-49] traveler style
  const STYLE_IDX = { solo:42, couple:43, family:44, group:45, business:46, senior:47, student:48, backpacker:49, leisure:43, cruise:44, extreme:45, "short-trip":42 };
  for (const s of product.travelStyles) if (STYLE_IDX[s] !== undefined) vec[STYLE_IDX[s]] = 1.0;

  // [50-55] provider quality
  const prov = product._provider;
  vec[50] = prov.tier / 5;
  vec[51] = prov.claimScore;
  vec[52] = product.rating / 5;
  vec[53] = Math.min(product.reviewCount / 5000, 1);
  vec[54] = product.isSponsored ? 0.8 : 0.2;
  vec[55] = product.freshnessScore;

  // [56-63] coverage amount (log-scaled bins)
  const logCov = Math.log10(product.coverageAmount + 1);
  const covBin = Math.min(Math.floor((logCov - 4) * 2), 7); // 10k-1M range
  if (covBin >= 0) vec[56 + Math.max(0, covBin)] = 1.0;

  // [64-127] deterministic noise for product uniqueness
  for (let i = 64; i < 128; i++) vec[i] = (r() - 0.5) * 0.6;

  return normalize(vec);
}

// ─── Product factory ──────────────────────────────────────────────────────────

function generateProducts(count = 96) {
  const products = [];
  let globalSeed = 99;

  for (let i = 0; i < count; i++) {
    const r         = seededRandom(globalSeed++);
    const type      = PRODUCT_TYPES[i % PRODUCT_TYPES.length];
    const provider  = PROVIDERS[Math.floor(r() * PROVIDERS.length)];
    const names     = NAMES_BY_TYPE[type];
    const nameIndex = Math.floor(r() * names.length);
    const [pMin, pMax] = PRICE_RANGE[type];
    const [cMin, cMax] = COVERAGE_RANGE[type];

    const pricePerDay    = Math.round((pMin + r() * (pMax - pMin)) * 100) / 100;
    const coverageAmount = Math.round((cMin + r() * (cMax - cMin)) / 1000) * 1000;
    const maxTripDays    = [7, 14, 21, 30, 45, 60, 90][Math.floor(r() * 7)];
    const rating         = Math.round((3.5 + r() * 1.5) * 10) / 10;
    const reviewCount    = Math.round(100 + r() * 4900);
    const isSponsored    = i < 6 && r() > 0.7;
    const freshnessScore = Math.round((0.5 + r() * 0.5) * 100) / 100;
    const badge          = BADGES[i % BADGES.length];

    const product = {
      id:             `ins_${String(i + 1).padStart(4, "0")}`,
      name:           `${names[nameIndex]} ${provider.name.split(" ")[0]}`,
      provider:       provider.name,
      type,
      pricePerDay,
      coverageAmount,
      maxTripDays,
      features:       FEATURES_BY_TYPE[type],
      destinations:   DESTINATIONS_BY_TYPE[type],
      travelStyles:   STYLES_BY_TYPE[type],
      rating,
      reviewCount,
      isSponsored,
      freshnessScore,
      badge,
      _provider:      provider, // internal, used for embedding
    };

    product.embedding = computeProductEmbedding(product, globalSeed++);
    products.push(product);
  }

  return products;
}

export const mockInsuranceProducts = generateProducts(96);
