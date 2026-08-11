import React from "react";

import { CHART } from "../../constants";
import "./charts.css";

/**
 * The confusion matrix is genuinely a table, so it is marked up as one — the
 * colour only encodes how much of the test set landed in each cell, and every
 * cell still states its count, its share and what kind of outcome it is.
 */
export default function ConfusionMatrix({ matrix, positiveLabel, negativeLabel }) {
  const cells = {
    true_negative: {
      value: matrix.true_negative,
      kind: "Correct",
      name: `True ${negativeLabel.toLowerCase()}`,
    },
    false_positive: {
      value: matrix.false_positive,
      kind: "Error",
      name: "False positive",
    },
    false_negative: {
      value: matrix.false_negative,
      kind: "Error",
      name: "False negative",
    },
    true_positive: {
      value: matrix.true_positive,
      kind: "Correct",
      name: `True ${positiveLabel.toLowerCase()}`,
    },
  };

  const total = Object.values(cells).reduce((sum, cell) => sum + cell.value, 0) || 1;
  const largest = Math.max(...Object.values(cells).map((cell) => cell.value), 1);

  const renderCell = (key) => {
    const cell = cells[key];
    const intensity = cell.value / largest;
    const step = Math.min(
      CHART.ramp.length - 1,
      Math.floor(intensity * CHART.ramp.length)
    );
    const isDark = step >= 2;

    return (
      <td
        key={key}
        className="confusion__cell"
        style={{
          background: CHART.ramp[step],
          color: isDark ? "#ffffff" : CHART.ink,
        }}
      >
        <span className="confusion__count">{cell.value.toLocaleString("en-US")}</span>
        <span className="confusion__share">{((cell.value / total) * 100).toFixed(1)}%</span>
        <span className="confusion__name">{cell.name}</span>
      </td>
    );
  };

  return (
    <div className="confusion">
      <table className="confusion__table">
        <caption className="confusion__caption">
          Rows are the actual outcome, columns are what the model predicted.
        </caption>
        <thead>
          <tr>
            <td className="confusion__corner" />
            <th scope="col">Predicted {negativeLabel.toLowerCase()}</th>
            <th scope="col">Predicted {positiveLabel.toLowerCase()}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Actual {negativeLabel.toLowerCase()}</th>
            {renderCell("true_negative")}
            {renderCell("false_positive")}
          </tr>
          <tr>
            <th scope="row">Actual {positiveLabel.toLowerCase()}</th>
            {renderCell("false_negative")}
            {renderCell("true_positive")}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
