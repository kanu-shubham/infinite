import React, { useState } from "react";
import HotelListingPage from "./features/hotels/HotelListingPage";
import PricingDashboard from "./features/pricing/PricingDashboard";
import "./App.css";

const TABS = [
  { id: "hotels", label: "Hotels" },
  { id: "pricing", label: "Dynamic Pricing" },
];

export default function App() {
  const [tab, setTab] = useState("pricing");
  return (
    <div className="app">
      <nav className="app-nav" aria-label="Primary">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`app-nav-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
            aria-pressed={tab === t.id}
          >
            {t.label}
          </button>
        ))}
      </nav>
      {tab === "hotels" ? <HotelListingPage /> : <PricingDashboard />}
    </div>
  );
}
