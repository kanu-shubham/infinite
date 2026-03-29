import React, { useCallback } from "react";
import "./Pagination.css";

/**
 * Generates the page number sequence with ellipsis gaps.
 *
 * Examples (siblings = 1 page either side of current):
 *   total=5,  current=3  → [1, 2, 3, 4, 5]
 *   total=10, current=1  → [1, 2, 3, '...', 10]
 *   total=10, current=5  → [1, '...', 4, 5, 6, '...', 10]
 *   total=10, current=9  → [1, '...', 8, 9, 10]
 */
function getPageNumbers(current, total) {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  const rangeStart = Math.max(2, current - 1);
  const rangeEnd   = Math.min(total - 1, current + 1);
  const pages      = [1];

  if (rangeStart > 2)       pages.push("...");
  for (let i = rangeStart; i <= rangeEnd; i++) pages.push(i);
  if (rangeEnd < total - 1) pages.push("...");
  pages.push(total);

  return pages;
}

export default function Pagination({ currentPage, totalPages, onPageChange }) {
  const pages = getPageNumbers(currentPage, totalPages);

  const go = useCallback(
    (page) => {
      if (page >= 1 && page <= totalPages && page !== currentPage) {
        onPageChange(page);
        // Scroll to top so the user sees the new page from the beginning
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    },
    [currentPage, totalPages, onPageChange]
  );

  if (totalPages <= 1) return null;

  return (
    <nav className="pagination" aria-label="Hotel results pages">
      {/* ── Prev ── */}
      <button
        className="pagination__btn pagination__btn--arrow"
        onClick={() => go(currentPage - 1)}
        disabled={currentPage === 1}
        aria-label="Previous page"
      >
        ‹
      </button>

      {/* ── Page numbers ── */}
      {pages.map((page, i) =>
        page === "..." ? (
          <span key={`ellipsis-${i}`} className="pagination__ellipsis">
            …
          </span>
        ) : (
          <button
            key={page}
            className={`pagination__btn${
              page === currentPage ? " pagination__btn--active" : ""
            }`}
            onClick={() => go(page)}
            aria-label={`Page ${page}`}
            aria-current={page === currentPage ? "page" : undefined}
          >
            {page}
          </button>
        )
      )}

      {/* ── Next ── */}
      <button
        className="pagination__btn pagination__btn--arrow"
        onClick={() => go(currentPage + 1)}
        disabled={currentPage === totalPages}
        aria-label="Next page"
      >
        ›
      </button>
    </nav>
  );
}
