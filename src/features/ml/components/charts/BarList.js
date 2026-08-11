import React from "react";
import "./charts.css";

/**
 * Horizontal bars for a single measure — feature importance, class balance,
 * cross-validation folds.
 *
 * One series means one colour: shading each bar by its own length would
 * double-encode the value and burn the only free channel. Bars are capped at
 * 14px, rounded at the data end and square at the baseline.
 */
export default function BarList({ items, formatValue, emptyMessage = "No data." }) {
  if (!items.length) {
    return <p className="chart-empty">{emptyMessage}</p>;
  }

  const maximum = Math.max(...items.map((item) => item.value), 0) || 1;

  return (
    <ul className="bar-list">
      {items.map((item) => {
        const share = Math.max((item.value / maximum) * 100, item.value > 0 ? 1.5 : 0);
        const formatted = formatValue ? formatValue(item.value) : item.value;

        return (
          <li className="bar-list__row" key={item.id}>
            <span
              className="bar-list__label"
              title={item.sublabel ? `${item.label} · ${item.sublabel}` : item.label}
            >
              {item.label}
              {item.sublabel && (
                <span className="bar-list__sublabel">{item.sublabel}</span>
              )}
            </span>

            <span className="bar-list__track">
              <span
                className="bar-list__bar"
                style={{ width: `${share}%` }}
                role="img"
                aria-label={`${item.label}: ${formatted}`}
              />
              <span className="bar-list__tooltip" role="tooltip">
                {item.label}
                {item.sublabel ? ` · ${item.sublabel}` : ""} — {formatted}
              </span>
            </span>

            <span className="bar-list__value">{formatted}</span>
          </li>
        );
      })}
    </ul>
  );
}
