import { useMemo, useState } from "react";
import { clock } from "../lib/format";

export interface Series {
  name: string;
  /** A chart token — var(--chart-1|2|3). Assigned in fixed order, never cycled. */
  color: string;
  values: Array<number | null>;
}

const PAD = { top: 8, right: 10, bottom: 16, left: 42 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const norm = v / mag;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * mag;
}

/**
 * Time-series line chart. One y-axis only — a second measure of a different
 * scale gets its own chart rather than a twin axis.
 */
export function LineChart({
  series,
  times,
  height = 148,
  format = (v: number) => String(Math.round(v)),
  unit = "",
  fixedMax,
}: {
  series: Series[];
  times: number[];
  height?: number;
  format?: (v: number) => string;
  unit?: string;
  fixedMax?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = height;

  const max = useMemo(() => {
    if (fixedMax != null) return fixedMax;
    let m = 0;
    for (const s of series)
      for (const v of s.values) if (v != null && isFinite(v) && v > m) m = v;
    return niceMax(m * 1.1);
  }, [series, fixedMax]);

  const n = times.length;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const x = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const y = (v: number) => PAD.top + plotH - (Math.min(v, max) / max) * plotH;

  const paths = useMemo(
    () =>
      series.map((s) => {
        // Nulls break the line rather than interpolating across a gap.
        let d = "";
        let pen = false;
        s.values.forEach((v, i) => {
          if (v == null || !isFinite(v)) {
            pen = false;
            return;
          }
          d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
          pen = true;
        });
        return d;
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [series, max, n],
  );

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  if (n === 0) {
    return (
      <p className="field__hint" style={{ padding: "var(--space-md) 0" }}>
        No samples yet — the chart fills once a run is producing traffic.
      </p>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ height: H }}
        role="img"
        aria-label={`${series.map((s) => s.name).join(", ")} over time`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const px = ((e.clientX - rect.left) / rect.width) * W;
          const i = Math.round(((px - PAD.left) / plotW) * (n - 1));
          setHover(Math.max(0, Math.min(n - 1, i)));
        }}
      >
        {ticks.map((t) => {
          const yy = PAD.top + plotH - t * plotH;
          return (
            <g key={t}>
              <line
                className="chart__grid"
                x1={PAD.left}
                x2={W - PAD.right}
                y1={yy}
                y2={yy}
              />
              <text className="chart__axis" x={PAD.left - 6} y={yy + 3} textAnchor="end">
                {format(t * max)}
              </text>
            </g>
          );
        })}

        {hover != null ? (
          <line
            className="chart__grid"
            x1={x(hover)}
            x2={x(hover)}
            y1={PAD.top}
            y2={PAD.top + plotH}
            style={{ stroke: "var(--color-neutral)" }}
          />
        ) : null}

        {series.map((s, si) => (
          <path
            key={s.name}
            className="chart__line"
            d={paths[si]}
            style={{ stroke: s.color, strokeWidth: 2 }}
          />
        ))}

        {hover != null
          ? series.map((s) => {
              const v = s.values[hover];
              if (v == null || !isFinite(v)) return null;
              return (
                <circle
                  key={s.name}
                  cx={x(hover)}
                  cy={y(v)}
                  r={4}
                  fill={s.color}
                  stroke="var(--chart-surface)"
                  strokeWidth={2}
                />
              );
            })
          : null}

        <text className="chart__axis" x={PAD.left} y={H - 4}>
          {clock(times[0])}
        </text>
        <text className="chart__axis" x={W - PAD.right} y={H - 4} textAnchor="end">
          {clock(times[n - 1])}
        </text>
      </svg>

      {hover != null ? (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: `${(x(hover) / W) * 100}%`,
            transform: `translateX(${hover > n / 2 ? "-104%" : "4%"})`,
            pointerEvents: "none",
            background: "var(--color-paper-4)",
            border: "1px solid var(--color-rule)",
            borderRadius: "var(--radius-sm)",
            padding: "var(--space-2xs) var(--space-xs)",
            fontFamily: "var(--font-display)",
            fontSize: "var(--text-2xs)",
            whiteSpace: "nowrap",
            boxShadow: "var(--shadow-card)",
            zIndex: 2,
          }}
        >
          <div className="dim">{clock(times[hover])}</div>
          {series.map((s) => (
            <div key={s.name} style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <i
                style={{
                  width: 8,
                  height: 2,
                  background: s.color,
                  display: "inline-block",
                  borderRadius: 999,
                }}
              />
              <span className="dim">{s.name}</span>
              <span style={{ marginLeft: "auto", color: "var(--color-ink)" }}>
                {s.values[hover] != null
                  ? `${format(s.values[hover] as number)}${unit}`
                  : "—"}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Legend. Always rendered for ≥2 series; a single series is named by the
 *  panel title instead, so no legend box. */
export function Legend({ series }: { series: Series[] }) {
  if (series.length < 2) return null;
  return (
    <div className="chart-legend">
      {series.map((s) => (
        <span key={s.name}>
          <i style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}

export interface Bar {
  label: string;
  value: number | null;
  color: string;
}

/** Categorical comparison. Bars are anchored to the baseline with rounded
 *  data-ends, separated by a surface gap, and directly labelled. */
export function BarChart({
  bars,
  height = 132,
  format = (v: number) => String(Math.round(v)),
  unit = "",
}: {
  bars: Bar[];
  height?: number;
  format?: (v: number) => string;
  unit?: string;
}) {
  const W = 360;
  const H = height;
  const pad = { top: 18, bottom: 22, left: 8, right: 8 };
  const plotH = H - pad.top - pad.bottom;
  const usable = bars.filter((b) => b.value != null && isFinite(b.value));

  if (usable.length === 0) {
    return (
      <p className="field__hint" style={{ padding: "var(--space-md) 0" }}>
        No completed requests to compare yet.
      </p>
    );
  }

  const max = niceMax(Math.max(...usable.map((b) => b.value as number)) * 1.15);
  const slot = (W - pad.left - pad.right) / bars.length;
  const barW = Math.min(64, slot - 2); // 2px surface gap between adjacent bars
  const r = 4;

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ height: H }}
      role="img"
      aria-label={bars
        .map((b) => `${b.label}: ${b.value != null ? format(b.value) : "no data"}`)
        .join(", ")}
    >
      <line
        className="chart__grid"
        x1={pad.left}
        x2={W - pad.right}
        y1={pad.top + plotH}
        y2={pad.top + plotH}
      />
      {bars.map((b, i) => {
        const cx = pad.left + slot * i + slot / 2;
        const x0 = cx - barW / 2;
        const h = b.value != null ? (Math.min(b.value, max) / max) * plotH : 0;
        const y0 = pad.top + plotH - h;
        const rr = Math.min(r, h);
        const d =
          h > 0
            ? `M${x0},${pad.top + plotH} L${x0},${y0 + rr} Q${x0},${y0} ${x0 + rr},${y0} L${x0 + barW - rr},${y0} Q${x0 + barW},${y0} ${x0 + barW},${y0 + rr} L${x0 + barW},${pad.top + plotH} Z`
            : "";
        return (
          <g key={b.label}>
            {h > 0 ? <path d={d} fill={b.color} /> : null}
            <text
              className="chart__axis"
              x={cx}
              y={y0 - 5}
              textAnchor="middle"
              style={{ fill: "var(--color-ink)" }}
            >
              {b.value != null ? `${format(b.value)}${unit}` : "—"}
            </text>
            <text className="chart__axis" x={cx} y={H - 6} textAnchor="middle">
              {b.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
