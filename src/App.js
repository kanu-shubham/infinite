import React, { useState } from "react";
import HotelListingPage from "./features/hotels/HotelListingPage";
import MlPipelinePage from "./features/ml/MlPipelinePage";
import "./App.css";

const VIEWS = [
  { id: "hotels", label: "Hotel Listing", component: HotelListingPage },
  { id: "ml", label: "ML Pipeline", component: MlPipelinePage },
];

export default function App() {
  const [view, setView] = useState("ml");

  const ActiveView = (VIEWS.find((item) => item.id === view) ?? VIEWS[0]).component;

  return (
    <div className="app">
      <nav className="app-nav" aria-label="Application areas">
        <span className="app-nav__brand">infinite</span>
        <div className="app-nav__links">
          {VIEWS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`app-nav__link${view === item.id ? " app-nav__link--active" : ""}`}
              aria-current={view === item.id ? "page" : undefined}
              onClick={() => setView(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </nav>

      <ActiveView />
    </div>
  );
}
