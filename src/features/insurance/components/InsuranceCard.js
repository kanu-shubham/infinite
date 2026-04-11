import React, { memo } from "react";
import { TYPE_LABELS, TYPE_COLORS } from "../data/mockInsuranceProducts";
import "./InsuranceCard.css";

const InsuranceCard = memo(function InsuranceCard({ product, rank }) {
  const typeColor = TYPE_COLORS[product.type] || "#6366f1";
  const matchPct  = Math.round((product.rankScore ?? 0) * 100);
  const featurePct =
    product.totalWantedFeatures > 0
      ? Math.round((product.featureMatchCount / product.totalWantedFeatures) * 100)
      : null;

  return (
    <article className="ins-card" style={{ "--type-color": typeColor }}>
      {/* ── Badges row ── */}
      <div className="ins-card__badges">
        <span className="ins-card__type-badge">
          {TYPE_LABELS[product.type] ?? product.type}
        </span>
        {product._sponsoredSlot && (
          <span className="ins-card__sponsored-badge">Sponsored</span>
        )}
        {product.badge && (
          <span className="ins-card__badge">{product.badge}</span>
        )}
      </div>

      {/* ── Header ── */}
      <div className="ins-card__header">
        <div>
          <h3 className="ins-card__name">{product.name}</h3>
          <p className="ins-card__provider">{product.provider}</p>
        </div>
        <div className="ins-card__price-block">
          <span className="ins-card__price">
            ${product.pricePerDay.toFixed(2)}
          </span>
          <span className="ins-card__price-unit">/ day</span>
        </div>
      </div>

      {/* ── Coverage & trip length ── */}
      <div className="ins-card__meta">
        <span className="ins-card__meta-item">
          <span className="ins-card__meta-label">Coverage</span>
          <span className="ins-card__meta-value">
            ${product.coverageAmount.toLocaleString()}
          </span>
        </span>
        <span className="ins-card__meta-item">
          <span className="ins-card__meta-label">Max Trip</span>
          <span className="ins-card__meta-value">{product.maxTripDays} days</span>
        </span>
        <span className="ins-card__meta-item">
          <span className="ins-card__meta-label">Rating</span>
          <span className="ins-card__meta-value">
            ★ {product.rating} <small>({product.reviewCount.toLocaleString()})</small>
          </span>
        </span>
      </div>

      {/* ── Feature chips ── */}
      <div className="ins-card__features">
        {product.features.slice(0, 4).map((f) => (
          <span key={f} className="ins-card__feature">{f}</span>
        ))}
        {product.features.length > 4 && (
          <span className="ins-card__feature ins-card__feature--more">
            +{product.features.length - 4} more
          </span>
        )}
      </div>

      {/* ── Match score bar ── */}
      <div className="ins-card__match">
        <div className="ins-card__match-labels">
          <span>Match score</span>
          <span className="ins-card__match-pct">{matchPct}%</span>
        </div>
        <div className="ins-card__match-bar">
          <div
            className="ins-card__match-fill"
            style={{ width: `${matchPct}%` }}
          />
        </div>
        {featurePct !== null && (
          <p className="ins-card__feature-match">
            {product.featureMatchCount}/{product.totalWantedFeatures} requested features covered
          </p>
        )}
      </div>

      {/* ── CTA ── */}
      <button className="ins-card__cta">Select Plan</button>

      {/* ── Rank indicator ── */}
      {rank <= 3 && (
        <div className="ins-card__rank">#{rank}</div>
      )}
    </article>
  );
});

export default InsuranceCard;
