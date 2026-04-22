import React, { useMemo } from "react";
import HotelListingPage from "./features/hotels/HotelListingPage";
import {
  MemoryProvider,
  createMemoryManager,
  LocalStorageAdapter,
} from "./memory";
import "./App.css";

export default function App() {
  const manager = useMemo(() => {
    const storage = new LocalStorageAdapter({ namespace: "infinite-memory" });
    return createMemoryManager({ storage });
  }, []);

  return (
    <MemoryProvider manager={manager}>
      <div className="app">
        <HotelListingPage />
      </div>
    </MemoryProvider>
  );
}
