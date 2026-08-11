import React, { useState } from "react";
import "./charts.css";

/**
 * Frame shared by every chart in the feature.
 *
 * It exists mostly for the table toggle: a chart that encodes a value in
 * position or colour needs a text equivalent, and giving all of them one
 * implementation means no chart can ship without one.
 */
export default function ChartCard({ title, subtitle, table, footnote, children }) {
  const [showTable, setShowTable] = useState(false);
  const canToggle = Boolean(table && table.rows.length);

  return (
    <figure className="chart-card">
      <figcaption className="chart-card__header">
        <div>
          <h4 className="chart-card__title">{title}</h4>
          {subtitle && <p className="chart-card__subtitle">{subtitle}</p>}
        </div>

        {canToggle && (
          <button
            type="button"
            className="chart-card__toggle"
            onClick={() => setShowTable((previous) => !previous)}
            aria-pressed={showTable}
          >
            {showTable ? "Chart" : "Table"}
          </button>
        )}
      </figcaption>

      <div className="chart-card__body">
        {showTable && canToggle ? (
          <div className="chart-card__table-wrap">
            <table className="chart-card__table">
              <thead>
                <tr>
                  {table.columns.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={column.align === "right" ? "is-right" : ""}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, index) => (
                  <tr key={row.id ?? index}>
                    {table.columns.map((column) => (
                      <td
                        key={column.key}
                        className={column.align === "right" ? "is-right" : ""}
                      >
                        {row[column.key]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          children
        )}
      </div>

      {footnote && <p className="chart-card__footnote">{footnote}</p>}
    </figure>
  );
}
