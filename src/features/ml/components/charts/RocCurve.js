import React, { useCallback, useMemo, useRef, useState } from "react";

import { CHART } from "../../constants";
import "./charts.css";

const VIEW = { width: 340, height: 300 };
const PAD = { top: 12, right: 16, bottom: 44, left: 46 };
const PLOT = {
  width: VIEW.width - PAD.left - PAD.right,
  height: VIEW.height - PAD.top - PAD.bottom,
};
const TICKS = [0, 0.25, 0.5, 0.75, 1];

/**
 * ROC curve for a classification run.
 *
 * One series, so no legend — the title says what is plotted. The chance
 * diagonal is drawn as a recessive solid hairline (dashing would read as a
 * threshold) and labelled in place. Hover snaps to the nearest threshold
 * point; arrow keys walk the same points for keyboard readers.
 */
export default function RocCurve({ points, auc }) {
  const svgRef = useRef(null);
  const [hoverIndex, setHoverIndex] = useState(null);

  const scaled = useMemo(
    () =>
      points.map((point) => ({
        ...point,
        x: PAD.left + point.fpr * PLOT.width,
        y: PAD.top + (1 - point.tpr) * PLOT.height,
      })),
    [points]
  );

  const linePath = useMemo(
    () => scaled.map((p, i) => `${i === 0 ? "M" : "L"}${p.x} ${p.y}`).join(" "),
    [scaled]
  );

  const areaPath = useMemo(() => {
    if (!scaled.length) return "";
    const baseline = PAD.top + PLOT.height;
    return `${linePath} L${scaled[scaled.length - 1].x} ${baseline} L${scaled[0].x} ${baseline} Z`;
  }, [linePath, scaled]);

  const handleMove = useCallback(
    (event) => {
      const svg = svgRef.current;
      if (!svg || !scaled.length) return;

      const bounds = svg.getBoundingClientRect();
      const viewX = ((event.clientX - bounds.left) / bounds.width) * VIEW.width;

      let nearest = 0;
      let smallest = Infinity;
      scaled.forEach((point, index) => {
        const distance = Math.abs(point.x - viewX);
        if (distance < smallest) {
          smallest = distance;
          nearest = index;
        }
      });
      setHoverIndex(nearest);
    },
    [scaled]
  );

  const handleKeyDown = useCallback(
    (event) => {
      if (!scaled.length) return;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      setHoverIndex((previous) => {
        const current = previous === null ? 0 : previous;
        const next = event.key === "ArrowRight" ? current + 1 : current - 1;
        return Math.min(Math.max(next, 0), scaled.length - 1);
      });
    },
    [scaled]
  );

  if (!points.length) return <p className="chart-empty">No ROC curve for this run.</p>;

  const active = hoverIndex === null ? null : scaled[hoverIndex];

  return (
    <div className="chart-frame">
      <svg
        ref={svgRef}
        className="chart-svg"
        viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
        role="img"
        aria-label={`ROC curve with an area under the curve of ${auc ?? "unknown"}`}
        tabIndex={0}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
        onFocus={() => setHoverIndex((previous) => previous ?? Math.floor(scaled.length / 2))}
        onBlur={() => setHoverIndex(null)}
        onKeyDown={handleKeyDown}
      >
        {TICKS.map((tick) => {
          const y = PAD.top + (1 - tick) * PLOT.height;
          const x = PAD.left + tick * PLOT.width;
          return (
            <g key={tick}>
              <line x1={PAD.left} y1={y} x2={PAD.left + PLOT.width} y2={y} stroke={CHART.grid} strokeWidth="1" />
              <text x={PAD.left - 8} y={y + 4} textAnchor="end" className="chart-svg__tick">
                {tick}
              </text>
              <text x={x} y={PAD.top + PLOT.height + 20} textAnchor="middle" className="chart-svg__tick">
                {tick}
              </text>
            </g>
          );
        })}

        {/* Chance line: a model with no signal traces this diagonal. */}
        <line
          x1={PAD.left}
          y1={PAD.top + PLOT.height}
          x2={PAD.left + PLOT.width}
          y2={PAD.top}
          stroke={CHART.axis}
          strokeWidth="1"
        />
        <text
          x={PAD.left + PLOT.width - 6}
          y={PAD.top + 26}
          textAnchor="end"
          className="chart-svg__annotation"
        >
          chance
        </text>

        <path d={areaPath} fill={CHART.seriesWash} />
        <path
          d={linePath}
          fill="none"
          stroke={CHART.series}
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {active && (
          <g>
            <line
              x1={active.x}
              y1={PAD.top}
              x2={active.x}
              y2={PAD.top + PLOT.height}
              stroke={CHART.axis}
              strokeWidth="1"
            />
            <circle cx={active.x} cy={active.y} r="5" fill={CHART.series} stroke={CHART.surface} strokeWidth="2" />
          </g>
        )}

        <line
          x1={PAD.left}
          y1={PAD.top + PLOT.height}
          x2={PAD.left + PLOT.width}
          y2={PAD.top + PLOT.height}
          stroke={CHART.axis}
          strokeWidth="1"
        />
        <text
          x={PAD.left + PLOT.width / 2}
          y={VIEW.height - 8}
          textAnchor="middle"
          className="chart-svg__axis-title"
        >
          False positive rate
        </text>
        <text
          transform={`translate(12 ${PAD.top + PLOT.height / 2}) rotate(-90)`}
          textAnchor="middle"
          className="chart-svg__axis-title"
        >
          True positive rate
        </text>
      </svg>

      {active && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(active.x / VIEW.width) * 100}%`,
            top: `${(active.y / VIEW.height) * 100}%`,
          }}
          role="tooltip"
        >
          <strong>TPR {active.tpr.toFixed(3)}</strong>
          <span>FPR {active.fpr.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
}
