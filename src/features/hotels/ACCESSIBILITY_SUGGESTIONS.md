# Accessibility Improvements — Hotel Listing Feature

> These are improvements that *should* be made before shipping to production.
> None are currently implemented in this codebase.

---

## 1. Live region announcements

Whenever filters are applied or the result count changes, screen reader users
get no feedback. Fix with an `aria-live` region:

```jsx
// Add once at the top of HotelListingPage
<div aria-live="polite" aria-atomic="true" className="sr-only">
  {!isLoading && `${totalCount} hotels found`}
</div>
```

The `sr-only` class makes it invisible but readable:
```css
.sr-only {
  position: absolute; width: 1px; height: 1px;
  padding: 0; overflow: hidden; clip: rect(0,0,0,0);
  white-space: nowrap; border: 0;
}
```

---

## 2. Semantic list markup

Hotel cards are in a `<div>` grid. Should be a `<ul>` so screen readers
announce "list of N items" and users can navigate with list shortcuts:

```jsx
// HotelList.js
<ul className="hotel-list__grid" role="list">
  {hotels.map(hotel => (
    <li key={hotel.id}>
      <HotelCard hotel={hotel} />
    </li>
  ))}
</ul>
```

---

## 3. Keyboard navigation on tab/toggle buttons

The tab bar and strategy toggle already use `<button>` (correct), but are
missing `aria-pressed` state so screen readers can't tell which is active:

```jsx
<button
  aria-pressed={activeTab === tab.id}
  onClick={() => setActiveTab(tab.id)}
>
  {tab.label}
</button>
```

Or treat them as a `role="tablist"` / `role="tab"` pattern if you want
full ARIA tab semantics with `aria-selected` and `aria-controls`.

---

## 4. Filter inputs — accessible labels

All filter inputs have `<label htmlFor>` which is good, but the "Clear Filters"
button has no accessible description of *what* it will clear:

```jsx
<button aria-label="Clear all active filters" onClick={onReset}>
  Clear Filters
</button>
```

---

## 5. Price badge on hotel image

The price badge is absolutely positioned over the image. Screen readers read
it out of order (image alt → price as separate unlabelled element). Fix:

```jsx
// In HotelCard
<article aria-label={`${hotel.name}, $${hotel.price} per night`}>
  ...
  <span className="hotel-card__price" aria-hidden="true">
    ${hotel.price}<small>/night</small>
  </span>
```

The `aria-label` on `<article>` gives the full context; `aria-hidden` on the
price badge prevents it being read twice.

---

## 6. Star rating

`StarRating` already has `aria-label="4.2 out of 5 stars"` — good.
But the raw star characters (★★★½☆) are still read by some screen readers.
Add `aria-hidden="true"` to the span and keep the aria-label on a wrapper:

```jsx
<span aria-label={`${rating} out of 5 stars`}>
  <span aria-hidden="true">{"★".repeat(fullStars)}{hasHalf && "½"}{"☆".repeat(emptyStars)}</span>
</span>
```

---

## 7. Focus management after filter change

When a filter is applied the list re-renders but focus stays on the select/input.
A keyboard user has to Tab all the way through the controls to reach the new
results. Consider moving focus to a "Results updated" live region or a skip link
target above the results:

```jsx
// Add a visually-hidden skip link
<a href="#hotel-results" className="sr-only skip-link">
  Skip to results
</a>

// Add id to the results container
<div id="hotel-results" tabIndex={-1}>
  <HotelList ... />
</div>
```

---

## 8. Loading skeleton accessibility

`HotelSkeletonGrid` already has `role="status"` and `aria-label` — good.
When the skeleton is replaced by real content, the live region (point 1 above)
announces the count automatically, so no extra work needed there.

---

## 9. Color contrast failures

| Element | Current color | Background | Ratio | Required |
|---------|--------------|-----------|-------|----------|
| Review count text | `#9ca3af` | `#fff` | 2.85:1 | 4.5:1 ❌ |
| Amenity tag text | `#374151` | `#f3f4f6` | 9.1:1 | 4.5:1 ✓ |
| Rating stars | `#f59e0b` | `#fff` | 2.83:1 | 3:1 (large) ≈ borderline |

Fix: bump review count and faint labels to `#6b7280` minimum.

---

## 10. Reduced motion

All CSS animations (shimmer, spinner, card hover scale) should be disabled
for users who have `prefers-reduced-motion: reduce` set. Already done in
`HotelSkeleton.css` — apply the same pattern to `LoadingSpinner.css` and
the hover `transform` in `HotelCard.css`:

```css
@media (prefers-reduced-motion: reduce) {
  .loading-spinner__circle { animation: none; border-top-color: #2563eb; }
  .hotel-card:hover         { transform: none; }
  .hotel-card:hover .hotel-card__image { transform: none; }
}
```

---

## 11. Image alt text

Currently `alt={hotel.name}` which is better than empty but still generic.
Ideal alt text describes what the image *shows*, not just the entity name:

```jsx
alt={`Exterior view of ${hotel.name} in ${hotel.location}`}
```

---

## 12. Search input — debounce feedback

The search input debounces 350ms. During that delay a screen reader user
gets no feedback that a search is in progress. Add `aria-busy`:

```jsx
<div role="search" aria-busy={isLoading}>
  <input ... />
</div>
```
