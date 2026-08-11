import React, { useCallback, useMemo, useRef, useState } from "react";

import { CHART } from "../../constants";
import "./charts.css";

const VIEW = { width: 340, height: 300 };
const PAD = { top: 12, right: 16, bottom: 44, left: 52 };
const PLOT = {
  width: VIEW.width - PAD.left - PAD.right,
  height: VIEW.height - PAD.top - PAD.bottom,
};
const TICK_COUNT = 4;

function niceTicks(minimum, maximum, count) {
  const step = (maximum - minimum) / count;
  return Array.from({ length: count + 1 }, (_, index) =>
    Math.round(minimum + step * index)
  );
}

/**
 * Actual vs predicted for a regression run.
 *
 * The 45° line is where a perfect model would put every point, so vertical
 * distance from it reads directly as error. Dense marks get a nearest-point
 * hover layer rather than pinpoint hit targets.
 */
export default function PredictionScatter({ points, unit = "" }) {
  const svgRef = useRef(null);
  const [hoverIndex, setHoverIndex] = useState(null);

  const { scaled, ticks } = useMemo(() => {
    if (!points.length) return { scaled: [], ticks: [] };

    const values = points.flatMap((point) => [point.actual, point.predicted]);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const padding = (rawMax - rawMin) * 0.05 || 1;
    const minimum = rawMin - padding;
    const maximum = rawMax + padding;
    const span = maximum - minimum || 1;

    return {
      scaled: points.map((point, index) => ({
        ...point,
        index,
        x: PAD.left + ((point.actual - minimum) / span) * PLOT.width,
        y: PAD.top + (1 - (point.predicted - minimum) / span) * PLOT.height,
      })),
      ticks: niceTicks(minimum, maximum, TICK_COUNT).map((value) => ({
        value,
        x: PAD.left + ((value - minimum) / span) * PLOT.width,
        y: PAD.top + (1 - (value - minimum) / span) * PLOT.height,
      })),
    };
  }, [points]);

  const handleMove = useCallback(
    (event) => {
      const svg = svgRef.current;
      if (!svg || !scaled.length) return;

      const bounds = svg.getBoundingClientRect();
      const viewX = ((event.clientX - bounds.left) / bounds.width) * VIEW.width;
      const viewY = ((event.clientY - bounds.top) / bounds.height) * VIEW.height;

      let nearest = null;
      let smallest = Infinity;
      scaled.forEach((point) => {
        const distance = (point.x - viewX) ** 2 + (point.y - viewY) ** 2;
        if (distance < smallest) {
          smallest = distance;
          nearest = point.index;
        }
      });
      // Only latch on when the cursor is genuinely near a mark.
      setHoverIndex(smallest <= 30 ** 2 ? nearest : null);
    },
    [scaled]
  );

  if (!points.length) return <p className="chart-empty">No predictions to plot.</p>;

  const active = hoverIndex === null ? null : scaled[hoverIndex];
  const diagonalStart = ticks[0];
  const diagonalEnd = ticks[ticks.length - 1];

  return (
    <div className="chart-frame">
      <svg
        ref={svgRef}
        className="chart-svg"
        viewBox={`0 0 ${VIEW.width} ${VIEW.height}`}
        role="img"
        aria-label={`Scatter of ${points.length} predicted values against their actual values`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {ticks.map((tick) => (
          <g key={tick.value}>
            <line
              x1={PAD.left}
              y1={tick.y}
              x2={PAD.left + PLOT.width}
              y2={tick.y}
              stroke={CHART.grid}
              strokeWidth="1"
            />
            <text x={PAD.left - 8} y={tick.y + 4} textAnchor="end" className="chart-svg__tick">
              {tick.value}
            </text>
            <text
              x={tick.x}
              y={PAD.top + PLOT.height + 20}
              textAnchor="middle"
              className="chart-svg__tick"
            >
              {tick.value}
            </text>
          </g>
        ))}

        {/* Perfect-prediction line. */}
        <line
          x1={diagonalStart.x}
          y1={diagonalStart.y}
          x2={diagonalEnd.x}
          y2={diagonalEnd.y}
          stroke={CHART.axis}
          strokeWidth="1"
        />
        <text
          x={diagonalEnd.x - 6}
          y={diagonalEnd.y + 18}
          textAnchor="end"
          className="chart-svg__annotation"
        >
          perfect fit
        </text>

        {scaled.map((point) => (
          <circle
            key={point.index}
            cx={point.x}
            cy={point.y}
            r="4"
            fill={CHART.series}
            fillOpacity={active && active.index === point.index ? 1 : 0.4}
          />
        ))}

        {active && (
          <circle
            cx={active.x}
            cy={active.y}
            r="5"
            fill={CHART.series}
            stroke={CHART.surface}
            strokeWidth="2"
          />
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
          Actual {unit}
        </text>
        <text
          transform={`translate(14 ${PAD.top + PLOT.height / 2}) rotate(-90)`}
          textAnchor="middle"
          className="chart-svg__axis-title"
        >
          Predicted {unit}
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
          <strong>Predicted {active.predicted.toFixed(2)}</strong>
          <span>Actual {active.actual.toFixed(2)}</span>
          <span>Error {(active.predicted - active.actual).toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}
