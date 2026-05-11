import React, { memo } from "react";
import useVirtualList from "../hooks/useVirtualList";
import useWindowVirtualizer from "../hooks/useWindowVirtualizer";
import LoadingSpinner from "../../../components/common/LoadingSpinner";
import ErrorMessage from "../../../components/common/ErrorMessage";
import "./VirtualHotelList.css";

const ITEM_HEIGHT = 170; // px — height of each compact row including gap

// Extracted to module scope — recreating these on every scroll tick caused
// style recalculations even though the values never changed.
const SPACER_STYLE = { position: "relative" };
const ITEM_WRAPPER_STYLE = {
  position: "absolute",
  width: "100%",
  height: ITEM_HEIGHT,
  padding: "0 0 12px",
};

// ---------------------------------------------------------------------------
// Compact card shared by both strategies
// memo: virtualItems array is recreated on every scroll tick, so without
// memo every visible card would re-render on scroll even though hotel data
// hasn't changed.
// ---------------------------------------------------------------------------
const CompactHotelCard = memo(function CompactHotelCard({ hotel }) {
  return (
    <div className="compact-card">
      <img
        className="compact-card__image"
        src={hotel.image}
        alt={hotel.name}
        loading="lazy"
      />
      <div className="compact-card__body">
        <div className="compact-card__top">
          <h3 className="compact-card__name">{hotel.name}</h3>
          <span className="compact-card__price">${hotel.price}<small>/night</small></span>
        </div>
        <p className="compact-card__location">{hotel.location}</p>
        <div className="compact-card__rating">
          <span className="compact-card__stars">{"★".repeat(Math.round(hotel.rating))}</span>
          <span className="compact-card__rating-value">{hotel.rating}</span>
          <span className="compact-card__reviews">({hotel.reviewCount} reviews)</span>
        </div>
        <div className="compact-card__amenities">
          {hotel.amenities.slice(0, 3).map((a) => (
            <span key={a} className="compact-card__amenity">{a}</span>
          ))}
        </div>
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Strategy A — container-based (DEFAULT)
// Scroll happens inside a fixed-height div; window is not involved.
// memo: prevents re-render when the parent switches strategy toggle or when
// unrelated state (e.g. the other tab's data) changes in HotelListingPage.
// ---------------------------------------------------------------------------
const ContainerVirtualList = memo(function ContainerVirtualList({ hotels }) {
  const { containerRef, virtualItems, totalHeight, startIndex, endIndex } =
    useVirtualList({ itemCount: hotels.length, itemHeight: ITEM_HEIGHT });

  return (
    <div className="virtual-list__meta">
      <span className="virtual-list__badge">
        Rendering {virtualItems.length} of {hotels.length} items
      </span>
      <div ref={containerRef} className="virtual-list__container">
        <div style={{ height: totalHeight, ...SPACER_STYLE }}>
          {virtualItems.map(({ index, offsetTop }) => (
            <div
              key={hotels[index].id}
              style={{ ...ITEM_WRAPPER_STYLE, top: offsetTop }}
            >
              <CompactHotelCard hotel={hotels[index]} />
            </div>
          ))}
        </div>
      </div>
      <p className="virtual-list__info">
        Visible rows: {startIndex}–{endIndex}
      </p>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Strategy B — window-based
// Scroll is tracked on window; list is positioned in normal document flow.
// memo: same reasoning as ContainerVirtualList.
// ---------------------------------------------------------------------------
const WindowVirtualList = memo(function WindowVirtualList({ hotels }) {
  const { listRef, virtualItems, totalHeight, startIndex, endIndex } =
    useWindowVirtualizer({ itemCount: hotels.length, itemHeight: ITEM_HEIGHT });

  return (
    <div className="virtual-list__meta">
      <span className="virtual-list__badge">
        Rendering {virtualItems.length} of {hotels.length} items
      </span>
      <div ref={listRef} style={{ height: totalHeight, ...SPACER_STYLE }}>
        {virtualItems.map(({ index, offsetTop }) => (
          <div
            key={hotels[index].id}
            style={{ ...ITEM_WRAPPER_STYLE, top: offsetTop }}
          >
            <CompactHotelCard hotel={hotels[index]} />
          </div>
        ))}
      </div>
      <p className="virtual-list__info">
        Visible rows: {startIndex}–{endIndex}
      </p>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Public component — selects strategy via prop
// ---------------------------------------------------------------------------
export default function VirtualHotelList({ hotels, isLoading, error, retry, strategy }) {
  if (isLoading) {
    return <LoadingSpinner size="medium" text="Loading virtualized list..." />;
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={retry} />;
  }

  if (hotels.length === 0) {
    return (
      <div className="virtual-list__empty">
        No hotels match your filters.
      </div>
    );
  }

  if (strategy === "window") {
    return <WindowVirtualList hotels={hotels} />;
  }

  return <ContainerVirtualList hotels={hotels} />;
}
