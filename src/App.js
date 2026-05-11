import React from "react";
import HotelListingPage from "./features/hotels/HotelListingPage";
import FeatureRating from "./features/featureRating/FeatureRating";
import "./App.css";

export default function App() {
  return (
    <div className="app">
      <HotelListingPage />
      <FeatureRating />
    </div>
  );
}
