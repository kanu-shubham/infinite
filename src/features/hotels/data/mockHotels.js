const HOTEL_NAMES = [
  "The Grand Palace", "Ocean View Resort", "Mountain Lodge", "City Center Inn",
  "Sunset Beach Hotel", "Royal Garden Suites", "Harbor Light Inn", "Pine Valley Resort",
  "The Sapphire Hotel", "Riverside Retreat", "Cloud Nine Suites", "Golden Sands Resort",
  "The Emerald Inn", "Skyline Tower Hotel", "Coral Bay Resort", "Maple Leaf Lodge",
  "The Diamond Hotel", "Serenity Springs", "Crescent Moon Inn", "Starlight Suites",
  "The Platinum Hotel", "Blue Lagoon Resort", "Ivory Tower Inn", "Willow Creek Lodge",
  "The Amber Hotel", "Paradise Cove Resort", "Silver Lake Inn", "Cedar Ridge Lodge",
  "The Onyx Hotel", "Moonrise Bay Resort", "Crystal Peak Inn", "Redwood Valley Lodge",
  "The Jasper Hotel", "Tidal Wave Resort", "Northwind Inn", "Birchwood Suites",
  "The Opal Hotel", "Sunflower Meadow Inn", "Stormwatch Lodge", "Driftwood Resort",
  "The Ruby Hotel", "Glacier Point Inn", "Foxglove Suites", "Sandstone Lodge",
  "The Topaz Hotel", "Whispering Pines Inn", "Lakeshore Resort", "Autumn Breeze Lodge"
];

const LOCATIONS = [
  "New York, NY", "Los Angeles, CA", "Miami, FL", "Chicago, IL",
  "San Francisco, CA", "Seattle, WA", "Austin, TX", "Denver, CO",
  "Boston, MA", "Nashville, TN", "Portland, OR", "San Diego, CA"
];

const AMENITIES = [
  "Free WiFi", "Pool", "Spa", "Gym", "Restaurant", "Bar",
  "Room Service", "Parking", "Pet Friendly", "Beach Access",
  "Business Center", "Concierge"
];

const IMAGES = [
  "https://images.unsplash.com/photo-1566073771259-6a8506099945?w=400&h=300&fit=crop",
  "https://images.unsplash.com/photo-1551882547-ff40c63fe5fa?w=400&h=300&fit=crop",
  "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=400&h=300&fit=crop",
  "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=400&h=300&fit=crop",
  "https://images.unsplash.com/photo-1571896349842-33c89424de2d?w=400&h=300&fit=crop",
  "https://images.unsplash.com/photo-1564501049412-61c2a3083791?w=400&h=300&fit=crop"
];

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateHotels(count = 48) {
  const random = seededRandom(42);
  return Array.from({ length: count }, (_, i) => {
    const id = i + 1;
    const rating = Math.round((2 + random() * 3) * 10) / 10;
    const price = Math.round(50 + random() * 450);
    const reviewCount = Math.round(10 + random() * 990);
    const amenityCount = 3 + Math.floor(random() * 6);
    const shuffled = [...AMENITIES].sort(() => random() - 0.5);

    return {
      id,
      name: HOTEL_NAMES[i % HOTEL_NAMES.length],
      location: LOCATIONS[Math.floor(random() * LOCATIONS.length)],
      price,
      rating,
      reviewCount,
      image: IMAGES[i % IMAGES.length],
      amenities: shuffled.slice(0, amenityCount),
      description: `Experience exceptional comfort and service at ${HOTEL_NAMES[i % HOTEL_NAMES.length]}. Located in a prime area with easy access to local attractions.`
    };
  });
}

export const mockHotels = generateHotels();
