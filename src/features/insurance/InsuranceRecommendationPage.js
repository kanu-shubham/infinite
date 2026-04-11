import React, { useState, useMemo } from "react";
import { mockInsuranceProducts } from "./data/mockInsuranceProducts";
import { mockUserProfiles }      from "./data/mockUserProfiles";
import { runRecommendationPipeline } from "./services/recommendationPipeline";
import UserProfileSelector  from "./components/UserProfileSelector";
import PipelineDebugPanel   from "./components/PipelineDebugPanel";
import InsuranceCard        from "./components/InsuranceCard";
import "./InsuranceRecommendationPage.css";

export default function InsuranceRecommendationPage() {
  const [selectedProfileId, setSelectedProfileId] = useState(mockUserProfiles[0].id);

  const selectedProfile = useMemo(
    () => mockUserProfiles.find((p) => p.id === selectedProfileId),
    [selectedProfileId],
  );

  // Run the full 4-stage pipeline whenever the user profile changes.
  // In production this would be an async API call; here it's synchronous JS.
  const pipelineResult = useMemo(
    () => runRecommendationPipeline(selectedProfile, mockInsuranceProducts),
    [selectedProfile],
  );

  const { recommendations, stages, totalLatencyMs } = pipelineResult;

  return (
    <div className="ins-page">
      {/* ── Header ── */}
      <header className="ins-page__header">
        <div className="ins-page__header-inner">
          <div>
            <h1 className="ins-page__title">Protect Your Trip</h1>
            <p className="ins-page__subtitle">
              Personalised travel insurance — powered by a two-tower ML ranking pipeline
            </p>
          </div>
          <div className="ins-page__expedia-badge">
            <span className="ins-page__expedia-dot" />
            Expedia Travel Protection
          </div>
        </div>
      </header>

      <div className="ins-page__content">
        {/* ── Trip context banner ── */}
        <div className="ins-page__trip-banner">
          <div className="ins-page__trip-info">
            <span className="ins-page__trip-label">Upcoming trip</span>
            <span className="ins-page__trip-dest">
              {selectedProfile.upcomingTrip.destination}
            </span>
            <span className="ins-page__trip-meta">
              {selectedProfile.upcomingTrip.durationDays} days ·{" "}
              {selectedProfile.upcomingTrip.tripType}
            </span>
          </div>
          <p className="ins-page__trip-desc">{selectedProfile.description}</p>
        </div>

        {/* ── User profile selector ── */}
        <UserProfileSelector
          profiles={mockUserProfiles}
          selectedId={selectedProfileId}
          onSelect={setSelectedProfileId}
        />

        {/* ── Pipeline debugger ── */}
        <PipelineDebugPanel stages={stages} totalLatencyMs={totalLatencyMs} />

        {/* ── Results header ── */}
        <div className="ins-page__results-header">
          <div>
            <h2 className="ins-page__results-title">
              Recommended for {selectedProfile.name.split(" ")[0]}
            </h2>
            <p className="ins-page__results-count">
              {recommendations.length} plans · ranked by match score
            </p>
          </div>
          <div className="ins-page__legend">
            {["adventure", "comprehensive", "cfar", "medical", "family", "senior", "student", "basic"].map(
              (t) => (
                <TypeDot key={t} type={t} />
              ),
            )}
          </div>
        </div>

        {/* ── Insurance cards grid ── */}
        <div className="ins-page__grid">
          {recommendations.map((product, i) => (
            <InsuranceCard key={product.id} product={product} rank={i + 1} />
          ))}
        </div>

        {/* ── Architecture note ── */}
        <div className="ins-page__arch-note">
          <h3>How the pipeline works</h3>
          <div className="arch-steps">
            <ArchStep
              num="1"
              title="User Tower"
              desc="A neural network encodes the traveler's profile (age, trip type, preferences, price sensitivity) into a 128-dimensional embedding vector."
            />
            <ArchStep
              num="2"
              title="ANN Index"
              desc="The user vector is compared against all pre-indexed product embeddings using cosine similarity (FAISS/ScaNN in production). Top candidates are retrieved in ~1 ms."
            />
            <ArchStep
              num="3"
              title="Ranking Model"
              desc="A heavier scoring function applies richer signals — price fit, coverage amount, feature overlap, trip length compatibility, and provider quality — to re-rank the top candidates."
            />
            <ArchStep
              num="4"
              title="Business Rules"
              desc="Post-processing enforces provider diversity (max 3 per provider), a freshness boost for recently updated products, and injects sponsored listings at fixed positions."
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function TypeDot({ type }) {
  const COLORS = {
    basic: "#6366f1", comprehensive: "#0ea5e9", adventure: "#f97316",
    medical: "#ef4444", cfar: "#8b5cf6", family: "#22c55e",
    senior: "#eab308", student: "#14b8a6",
  };
  const LABELS = {
    basic: "Basic", comprehensive: "Comprehensive", adventure: "Adventure",
    medical: "Medical", cfar: "Cancel Any", family: "Family",
    senior: "Senior", student: "Student",
  };
  return (
    <span className="type-dot">
      <span className="type-dot__circle" style={{ background: COLORS[type] }} />
      <span className="type-dot__label">{LABELS[type]}</span>
    </span>
  );
}

function ArchStep({ num, title, desc }) {
  return (
    <div className="arch-step">
      <div className="arch-step__num">{num}</div>
      <div className="arch-step__body">
        <strong>{title}</strong>
        <p>{desc}</p>
      </div>
    </div>
  );
}
