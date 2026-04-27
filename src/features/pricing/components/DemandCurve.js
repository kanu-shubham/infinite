import React, { useMemo } from "react";

const PAD = { top: 16, right: 16, bottom: 36, left: 48 };

export default function DemandCurve({ curve, finalPrice, optimalPrice, basePrice }) {
  const dims = { w: 520, h: 240 };

  const { paths, axis } = useMemo(() => {
    if (!curve || !curve.length) return { paths: null, axis: null };
    const xs = curve.map((p) => p.price);
    const yUnits = curve.map((p) => p.expectedUnits);
    const yRev = curve.map((p) => p.expectedRevenue);
    const yLow = curve.map((p) => p.unitsLow);
    const yHigh = curve.map((p) => p.unitsHigh);

    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const uMax = Math.max(...yHigh) * 1.1;
    const rMax = Math.max(...yRev) * 1.1;

    const xScale = (v) =>
      PAD.left + ((v - xMin) / (xMax - xMin || 1)) * (dims.w - PAD.left - PAD.right);
    const uScale = (v) =>
      dims.h - PAD.bottom - (v / (uMax || 1)) * (dims.h - PAD.top - PAD.bottom);
    const rScale = (v) =>
      dims.h - PAD.bottom - (v / (rMax || 1)) * (dims.h - PAD.top - PAD.bottom);

    const buildPath = (ys, scale) =>
      ys
        .map((v, i) => `${i === 0 ? "M" : "L"}${xScale(xs[i]).toFixed(1)},${scale(v).toFixed(1)}`)
        .join(" ");

    const bandPath =
      yLow
        .map((v, i) => `${i === 0 ? "M" : "L"}${xScale(xs[i]).toFixed(1)},${uScale(v).toFixed(1)}`)
        .join(" ") +
      " " +
      yHigh
        .slice()
        .reverse()
        .map((v, i) => {
          const idx = yHigh.length - 1 - i;
          return `L${xScale(xs[idx]).toFixed(1)},${uScale(v).toFixed(1)}`;
        })
        .join(" ") +
      " Z";

    return {
      paths: {
        units: buildPath(yUnits, uScale),
        revenue: buildPath(yRev, rScale),
        band: bandPath,
      },
      axis: { xMin, xMax, uMax, rMax, xScale, uScale, rScale },
    };
  }, [curve]);

  if (!paths) {
    return <div className="dc-empty">No demand data yet — record some bookings to fit the model.</div>;
  }

  const markers = [
    { value: basePrice, color: "#888", label: "base" },
    { value: optimalPrice, color: "#2c7", label: "optimal" },
    { value: finalPrice, color: "#e63", label: "shown" },
  ].filter((m) => m.value != null);

  return (
    <svg width={dims.w} height={dims.h} className="demand-curve" role="img">
      <rect x={0} y={0} width={dims.w} height={dims.h} fill="#fafafa" />
      <path d={paths.band} fill="rgba(80,140,220,0.18)" />
      <path d={paths.units} stroke="#2466d1" strokeWidth={2} fill="none" />
      <path d={paths.revenue} stroke="#1f9d55" strokeWidth={2} fill="none" strokeDasharray="4 3" />

      {markers.map((m) => {
        const x = axis.xScale(m.value);
        return (
          <g key={m.label}>
            <line x1={x} x2={x} y1={PAD.top} y2={dims.h - PAD.bottom} stroke={m.color} strokeDasharray="2 3" />
            <text x={x + 3} y={PAD.top + 10} fill={m.color} fontSize="11">
              {m.label} ${Number(m.value).toFixed(0)}
            </text>
          </g>
        );
      })}

      <line x1={PAD.left} y1={dims.h - PAD.bottom} x2={dims.w - PAD.right} y2={dims.h - PAD.bottom} stroke="#999" />
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={dims.h - PAD.bottom} stroke="#999" />
      <text x={PAD.left} y={dims.h - 10} fontSize="11" fill="#444">
        price ${axis.xMin.toFixed(0)} → ${axis.xMax.toFixed(0)}
      </text>
      <text x={6} y={PAD.top + 4} fontSize="11" fill="#2466d1">
        units (solid, ±1σ band)
      </text>
      <text x={6} y={PAD.top + 18} fontSize="11" fill="#1f9d55">
        revenue (dashed)
      </text>
    </svg>
  );
}
