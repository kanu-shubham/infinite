import React, { useState } from "react";
import "./PipelineDebugPanel.css";

const STAGE_ICONS = ["🧠", "🔍", "📊", "⚖️"];

const STAGE_KEYS = ["userTower", "annRetrieval", "rankingModel", "businessRules"];

const STAGE_LABELS = {
  userTower:     "User Tower",
  annRetrieval:  "ANN Index",
  rankingModel:  "Ranking Model",
  businessRules: "Business Rules",
};

const STAGE_SUBTITLES = {
  userTower:     "Neural net → 128-dim user embedding",
  annRetrieval:  "FAISS/ScaNN → top candidates in ~1ms",
  rankingModel:  "Heavy neural net → re-score & pick top 20",
  businessRules: "Diversity · Freshness · Ads injection",
};

export default function PipelineDebugPanel({ stages, totalLatencyMs }) {
  const [open, setOpen] = useState(true);

  if (!stages) return null;

  return (
    <div className="pipeline-panel">
      <button
        className="pipeline-panel__toggle"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="pipeline-panel__toggle-label">
          Pipeline Debugger
          <span className="pipeline-panel__total">
            {totalLatencyMs.toFixed(2)} ms total
          </span>
        </span>
        <span className="pipeline-panel__chevron">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="pipeline-panel__body">
          <div className="pipeline-stages">
            {STAGE_KEYS.map((key, idx) => {
              const stage = stages[key];
              if (!stage) return null;
              return (
                <React.Fragment key={key}>
                  <div className="pipeline-stage">
                    <div className="pipeline-stage__icon">{STAGE_ICONS[idx]}</div>
                    <div className="pipeline-stage__content">
                      <div className="pipeline-stage__header">
                        <span className="pipeline-stage__label">
                          Stage {idx + 1} — {STAGE_LABELS[key]}
                        </span>
                        <span className="pipeline-stage__ms">
                          {stage.latencyMs.toFixed(3)} ms
                        </span>
                      </div>
                      <p className="pipeline-stage__subtitle">
                        {STAGE_SUBTITLES[key]}
                      </p>
                      <p className="pipeline-stage__desc">{stage.description}</p>

                      {/* Stage-specific detail rows */}
                      {key === "userTower" && (
                        <div className="pipeline-stage__detail">
                          <DetailPill label="Dims" value={stage.embeddingDim} />
                        </div>
                      )}
                      {key === "annRetrieval" && (
                        <div className="pipeline-stage__detail">
                          <DetailPill label="Catalogue" value={`${stage.totalProducts} products`} />
                          <DetailPill label="Retrieved"  value={stage.retrieved} />
                          <DetailPill label="Top sim"    value={stage.topCandidateScore} />
                        </div>
                      )}
                      {key === "rankingModel" && (
                        <div className="pipeline-stage__detail">
                          <DetailPill label="Input"     value={`${stage.inputCount} candidates`} />
                          <DetailPill label="Output"    value={`${stage.outputCount} ranked`} />
                          <DetailPill label="Top score" value={stage.topRankScore} />
                        </div>
                      )}
                      {key === "businessRules" && stage.appliedRules?.length > 0 && (
                        <ul className="pipeline-stage__rules">
                          {stage.appliedRules.map((r) => (
                            <li key={r}>{r}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>
                  {idx < STAGE_KEYS.length - 1 && (
                    <div className="pipeline-arrow">↓</div>
                  )}
                </React.Fragment>
              );
            })}
          </div>

          <div className="pipeline-panel__footer">
            <span>Total pipeline latency:</span>
            <strong>{totalLatencyMs.toFixed(2)} ms</strong>
            <span className="pipeline-panel__note">
              (in production: ~3–5 ms with GPU inference &amp; ANN index)
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function DetailPill({ label, value }) {
  return (
    <span className="detail-pill">
      <span className="detail-pill__label">{label}</span>
      <span className="detail-pill__value">{value}</span>
    </span>
  );
}
