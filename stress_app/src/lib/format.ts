/** Formatting helpers. Everything numeric in the UI goes through here so the
 *  tables stay column-aligned and no raw float leaks into the interface. */

export function ms(v: number | null | undefined, digits = 0): string {
  if (v == null || !isFinite(v)) return "—";
  if (v >= 10_000) return `${(v / 1000).toFixed(1)}s`;
  return `${v.toFixed(digits)}ms`;
}

export function secs(v: number | null | undefined): string {
  if (v == null || !isFinite(v)) return "—";
  const total = Math.floor(v / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v == null || !isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function pct(v: number | null | undefined, digits = 0): string {
  if (v == null || !isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function bytes(chars: number): string {
  if (chars < 1024) return `${chars} ch`;
  return `${(chars / 1024).toFixed(1)}k ch`;
}

export function clock(t: number): string {
  const d = new Date(t);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

/** Nearest-rank percentile over an unsorted array. Returns null when empty. */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1),
  );
  return sorted[idx];
}

export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}
