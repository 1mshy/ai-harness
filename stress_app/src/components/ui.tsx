import { useState, type ReactNode } from "react";
import type { Difficulty } from "../lib/types";

export function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string;
  hint?: ReactNode;
  error?: string | null;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
      {error ? (
        <span className="field__error">{error}</span>
      ) : hint ? (
        <span className="field__hint">{hint}</span>
      ) : null}
    </label>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="switch__track" aria-hidden="true">
        <span className="switch__thumb" />
      </span>
      <span className="switch__label">{label}</span>
    </label>
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  const lo = min ?? Number.NEGATIVE_INFINITY;
  const hi = max ?? Number.POSITIVE_INFINITY;

  // While a typed value is still out of range it lives here instead of being
  // clamped on the spot — otherwise the first keystroke of "500" into a
  // min-200 field snaps to 200 and the rest of the digits land on that.
  const [draft, setDraft] = useState<string | null>(null);

  const commit = () => {
    if (draft === null) return;
    const n = Number(draft);
    setDraft(null);
    if (draft.trim() !== "" && Number.isFinite(n)) {
      const clamped = Math.min(hi, Math.max(lo, n));
      if (clamped !== value) onChange(clamped);
    }
  };

  return (
    <input
      className="input num"
      type="number"
      value={draft ?? (Number.isFinite(value) ? String(value) : "0")}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => {
        const raw = e.target.value;
        const n = Number(raw);
        if (raw.trim() !== "" && Number.isFinite(n) && n >= lo && n <= hi) {
          setDraft(null);
          onChange(n);
        } else {
          setDraft(raw);
        }
      }}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          commit();
        } else if (e.key === "Escape") {
          setDraft(null);
        }
      }}
    />
  );
}

/** Slider paired with a live numeric readout — the number is editable too, so
 *  precise values don't require dragging. */
export function SliderInput({
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <div className="slider-row">
      <input
        className="slider"
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
      <NumberInput
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
      />
    </div>
  );
}

export function Badge({
  kind,
  children,
}: {
  kind: Difficulty | "ok" | "fail" | "muted";
  children: ReactNode;
}) {
  return <span className={`badge badge--${kind}`}>{children}</span>;
}

export function Stat({
  k,
  v,
  unit,
  sub,
  tone,
}: {
  k: string;
  v: string;
  unit?: string;
  sub?: string;
  tone?: "accent" | "ok" | "fail";
}) {
  return (
    <div className={`stat${tone ? ` stat--${tone}` : ""}`}>
      <div className="stat__k">{k}</div>
      <div className="stat__v">
        {v}
        {unit ? <small>{unit}</small> : null}
      </div>
      {sub ? <div className="stat__sub">{sub}</div> : null}
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty__title">{title}</div>
      <p className="empty__hint">{hint}</p>
    </div>
  );
}

export function Panel({
  title,
  actions,
  children,
  flush,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className="panel">
      <header className="panel__head">
        <h2 className="panel__title">{title}</h2>
        {actions}
      </header>
      <div className={flush ? undefined : "panel__body"}>{children}</div>
    </section>
  );
}
