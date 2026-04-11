import React, { useState } from "react";
import HotelListingPage from "./features/hotels/HotelListingPage";
import InsuranceRecommendationPage from "./features/insurance/InsuranceRecommendationPage";
import "./App.css";

const PAGES = [
  { id: "insurance", label: "Travel Insurance" },
  { id: "hotels",    label: "Hotels" },
];

export default function App() {
  const [activePage, setActivePage] = useState("insurance");

  return (
    <div className="app">
      <nav className="app-nav">
        <div className="app-nav__inner">
          <span className="app-nav__brand">Expedia</span>
          <div className="app-nav__tabs">
            {PAGES.map((page) => (
              <button
                key={page.id}
                className={`app-nav__tab${activePage === page.id ? " app-nav__tab--active" : ""}`}
                onClick={() => setActivePage(page.id)}
              >
                {page.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {activePage === "insurance" && <InsuranceRecommendationPage />}
      {activePage === "hotels"    && <HotelListingPage />}
    </div>
  );
}
