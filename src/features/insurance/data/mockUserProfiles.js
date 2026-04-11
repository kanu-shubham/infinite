/**
 * Mock traveler personas used to drive the recommendation pipeline.
 * Each profile is fed into the User Tower to produce a 128-dim embedding.
 *
 * Fields mirror the semantic dimensions of the product embedding space so
 * the dot-product similarity is meaningful.
 */

export const mockUserProfiles = [
  {
    id:           "user_sarah",
    name:         "Sarah Chen",
    avatar:       "SC",
    age:          32,
    tagline:      "Adventure Seeker",
    description:  "Solo traveler heading to Thailand & Nepal for 14 days of trekking and water sports.",
    upcomingTrip: { destination: "Southeast Asia", durationDays: 14, tripType: "adventure" },
    // Signals fed to User Tower
    preferences: {
      types:       ["adventure", "comprehensive"],
      features:    ["Medical Emergency", "Adventure Sports", "Evacuation", "Search & Rescue"],
      destinations:["international", "asia", "adventure"],
      styles:      ["solo", "backpacker", "extreme"],
      maxPricePerDay: 15,
      minCoverage:    100_000,
    },
  },
  {
    id:           "user_mike",
    name:         "Mike Torres",
    avatar:       "MT",
    age:          45,
    tagline:      "Frequent Business Flyer",
    description:  "Senior manager flying to Europe 3× per quarter. Values trip cancellation and schedule flexibility.",
    upcomingTrip: { destination: "London & Paris", durationDays: 7, tripType: "business" },
    preferences: {
      types:       ["cfar", "comprehensive"],
      features:    ["Cancel For Any Reason", "Trip Cancellation", "Travel Delay", "Rental Car"],
      destinations:["international", "schengen"],
      styles:      ["business", "leisure"],
      maxPricePerDay: 20,
      minCoverage:    50_000,
    },
  },
  {
    id:           "user_johnson",
    name:         "The Johnson Family",
    avatar:       "JF",
    age:          40,
    tagline:      "Family Vacation Planner",
    description:  "Two adults + two kids (7 & 10) heading to Orlando for 10 days. Needs child coverage and baggage protection.",
    upcomingTrip: { destination: "Orlando, FL", durationDays: 10, tripType: "family" },
    preferences: {
      types:       ["family", "comprehensive"],
      features:    ["Trip Cancellation", "Child Medical", "Baggage Loss", "Travel Delay", "Pet Care Coverage"],
      destinations:["domestic"],
      styles:      ["family", "leisure"],
      maxPricePerDay: 10,
      minCoverage:    30_000,
    },
  },
  {
    id:           "user_emma",
    name:         "Emma Larsson",
    avatar:       "EL",
    age:          23,
    tagline:      "Gap Year Backpacker",
    description:  "Budget traveler spending 90 days across Southeast Asia. Needs cheap, long-duration medical coverage.",
    upcomingTrip: { destination: "Southeast Asia", durationDays: 90, tripType: "backpacker" },
    preferences: {
      types:       ["student", "medical"],
      features:    ["Medical Emergency", "Trip Cancellation", "Baggage Loss", "Electronic Equipment"],
      destinations:["international", "asia", "backpacker"],
      styles:      ["student", "backpacker", "solo"],
      maxPricePerDay: 4,
      minCoverage:    20_000,
    },
  },
  {
    id:           "user_robert",
    name:         "Robert & Helen Kim",
    avatar:       "RK",
    age:          68,
    tagline:      "Retired Cruise Enthusiasts",
    description:  "Retired couple taking a 21-day Caribbean cruise. Robert has a pre-existing heart condition.",
    upcomingTrip: { destination: "Caribbean Cruise", durationDays: 21, tripType: "cruise" },
    preferences: {
      types:       ["senior", "medical"],
      features:    ["Medical Emergency", "Pre-existing Conditions", "Medical Repatriation", "Evacuation", "24/7 Medical Support"],
      destinations:["cruise", "international"],
      styles:      ["senior", "cruise"],
      maxPricePerDay: 18,
      minCoverage:    200_000,
    },
  },
  {
    id:           "user_alex",
    name:         "Alex Rivera",
    avatar:       "AR",
    age:          28,
    tagline:      "Budget Solo Traveler",
    description:  "Digital nomad visiting Bali for 30 days. Wants basic coverage at the lowest possible price.",
    upcomingTrip: { destination: "Bali, Indonesia", durationDays: 30, tripType: "leisure" },
    preferences: {
      types:       ["basic", "student"],
      features:    ["Trip Cancellation", "Baggage Loss", "Travel Delay", "24/7 Assistance"],
      destinations:["international", "asia"],
      styles:      ["solo", "leisure"],
      maxPricePerDay: 4,
      minCoverage:    10_000,
    },
  },
];
