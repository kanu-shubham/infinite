"use strict";

// In-memory store. Production would back this with Postgres + a feature store;
// the interface stays the same so the rest of the app doesn't need to change.

class Repository {
  constructor() {
    this.hotels = new Map();
    this.feedback = []; // { hotelId, price, units, booked, propensity, context, decisionId, ts }
    this.audit = []; // { decisionId, hotelId, proposedPrice, finalPrice, adjustments, model, ts }
    this.experiments = new Map(); // expId -> { variants: [{name, weight}], status }
  }

  upsertHotel(hotel) {
    this.hotels.set(hotel.id, hotel);
  }

  getHotel(id) {
    return this.hotels.get(id);
  }

  listHotels() {
    return Array.from(this.hotels.values());
  }

  recordFeedback(entry) {
    const row = { ...entry, ts: entry.ts || Date.now() };
    this.feedback.push(row);
    return row;
  }

  feedbackForHotel(hotelId) {
    return this.feedback.filter((f) => f.hotelId === hotelId);
  }

  recordAudit(entry) {
    const row = { ...entry, ts: entry.ts || Date.now() };
    this.audit.push(row);
    if (this.audit.length > 5000) this.audit.shift();
    return row;
  }

  recentAudit(limit = 50) {
    return this.audit.slice(-limit).reverse();
  }

  setExperiment(id, def) {
    this.experiments.set(id, { id, ...def });
  }

  getExperiment(id) {
    return this.experiments.get(id);
  }

  listExperiments() {
    return Array.from(this.experiments.values());
  }
}

module.exports = { Repository };
