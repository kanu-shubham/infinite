import React from "react";

import BarList from "./charts/BarList";
import ChartCard from "./charts/ChartCard";
import { formatCount, humanize } from "../utils/format";
import "./DatasetPanel.css";

const PREVIEW_COLUMNS = [
  "booking_id",
  "hotel",
  "market_segment",
  "deposit_type",
  "lead_time",
  "adr",
  "special_requests",
  "is_canceled",
];

function StatTile({ label, value, note }) {
  return (
    <div className="stat-tile">
      <span className="stat-tile__label">{label}</span>
      <span className="stat-tile__value">{value}</span>
      {note && <span className="stat-tile__note">{note}</span>}
    </div>
  );
}

/**
 * What the pipeline is about to train on: shape, target balance, where the
 * holes are, and a slice of raw rows.
 */
export default function DatasetPanel({ dataset }) {
  const { profile, preview } = dataset;
  const distribution = profile.target_distribution;

  const missingEntries = Object.entries(profile.missing_values);
  const missingItems = missingEntries.map(([column, count]) => ({
    id: column,
    label: humanize(column),
    value: count,
  }));

  const classItems =
    distribution.kind === "classes"
      ? distribution.classes.map((entry) => ({
          id: String(entry.value),
          label: entry.label,
          sublabel: `${(entry.share * 100).toFixed(1)}%`,
          value: entry.count,
        }))
      : [];

  return (
    <div className="dataset-panel">
      <div className="dataset-panel__stats">
        <StatTile label="Rows" value={formatCount(profile.rows)} note="synthetic bookings" />
        <StatTile
          label="Features"
          value={profile.feature_count}
          note={`${profile.numeric_features.length} numeric · ${profile.categorical_features.length} categorical`}
        />
        <StatTile
          label="Target"
          value={profile.target.column}
          note={`${profile.target.task} · ${profile.target.label.toLowerCase()}`}
        />
        <StatTile
          label="Missing values"
          value={formatCount(
            missingEntries.reduce((total, [, count]) => total + count, 0)
          )}
          note={`across ${missingEntries.length} column${missingEntries.length === 1 ? "" : "s"}`}
        />
      </div>

      <div className="dataset-panel__charts">
        {distribution.kind === "classes" ? (
          <ChartCard
            title="Target balance"
            subtitle={profile.target.description}
            footnote="An imbalanced target is why the classifiers default to balanced class weights."
            table={{
              columns: [
                { key: "label", label: "Class" },
                { key: "count", label: "Rows", align: "right" },
                { key: "share", label: "Share", align: "right" },
              ],
              rows: distribution.classes.map((entry) => ({
                id: entry.value,
                label: entry.label,
                count: formatCount(entry.count),
                share: `${(entry.share * 100).toFixed(1)}%`,
              })),
            }}
          >
            <BarList items={classItems} formatValue={formatCount} />
          </ChartCard>
        ) : (
          <ChartCard
            title="Target distribution"
            subtitle={profile.target.description}
          >
            <dl className="dataset-panel__summary">
              {["min", "median", "mean", "max", "std"].map((key) => (
                <div key={key} className="dataset-panel__summary-row">
                  <dt>{key.toUpperCase()}</dt>
                  <dd>{distribution[key]}</dd>
                </div>
              ))}
            </dl>
          </ChartCard>
        )}

        <ChartCard
          title="Missing values by column"
          subtitle="Handled inside the pipeline by median and most-frequent imputers."
          table={{
            columns: [
              { key: "column", label: "Column" },
              { key: "count", label: "Missing rows", align: "right" },
            ],
            rows: missingItems.map((item) => ({
              id: item.id,
              column: item.label,
              count: formatCount(item.value),
            })),
          }}
        >
          <BarList
            items={missingItems}
            formatValue={formatCount}
            emptyMessage="No missing values in this extract."
          />
        </ChartCard>
      </div>

      <section className="dataset-panel__preview">
        <h3 className="dataset-panel__preview-title">Sample rows</h3>
        <div className="dataset-panel__table-wrap">
          <table className="dataset-panel__table">
            <thead>
              <tr>
                {PREVIEW_COLUMNS.map((column) => (
                  <th key={column} scope="col">
                    {humanize(column)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.map((row) => (
                <tr key={row.booking_id}>
                  {PREVIEW_COLUMNS.map((column) => (
                    <td key={column}>
                      {row[column] === null || row[column] === undefined ? (
                        <span className="dataset-panel__null">null</span>
                      ) : (
                        String(row[column])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
