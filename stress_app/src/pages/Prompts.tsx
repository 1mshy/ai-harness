import { useMemo, useState } from "react";
import { useStore } from "../lib/store";
import { ALL_PROMPTS, CATEGORIES, PROMPT_COUNTS } from "../lib/prompts";
import type { Difficulty } from "../lib/types";
import { num } from "../lib/format";
import { Badge, Empty, Panel, Stat } from "../components/ui";

const DIFFS: Array<Difficulty | "all"> = ["all", "easy", "medium", "hard"];

export default function Prompts() {
  const enabled = useStore((s) => s.enabled);
  const togglePrompt = useStore((s) => s.togglePrompt);
  const setPromptsEnabled = useStore((s) => s.setPromptsEnabled);
  const mix = useStore((s) => s.config.mix);

  const [q, setQ] = useState("");
  const [diff, setDiff] = useState<Difficulty | "all">("all");
  const [cat, setCat] = useState("all");
  const [preview, setPreview] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return ALL_PROMPTS.filter((p) => {
      if (diff !== "all" && p.difficulty !== diff) return false;
      if (cat !== "all" && p.category !== cat) return false;
      if (!needle) return true;
      return (
        p.title.toLowerCase().includes(needle) ||
        p.id.toLowerCase().includes(needle) ||
        p.category.toLowerCase().includes(needle) ||
        p.text.toLowerCase().includes(needle)
      );
    });
  }, [q, diff, cat]);

  const armedCount = useMemo(
    () =>
      ALL_PROMPTS.filter(
        (p) => enabled[p.id] !== false && mix[p.difficulty] > 0,
      ).length,
    [enabled, mix],
  );

  const previewed = preview ? ALL_PROMPTS.find((p) => p.id === preview) : null;

  const meanChars = useMemo(() => {
    const byDiff = (d: Difficulty) => {
      const items = ALL_PROMPTS.filter((p) => p.difficulty === d);
      if (!items.length) return 0;
      return Math.round(
        items.reduce((a, b) => a + b.text.length, 0) / items.length,
      );
    };
    return { easy: byDiff("easy"), medium: byDiff("medium"), hard: byDiff("hard") };
  }, []);

  if (ALL_PROMPTS.length === 0) {
    return (
      <Empty
        title="Corpus is empty"
        hint="No JSON files were found under src/data/prompts. Drop a file shaped { difficulty, prompts: [...] } in there and restart the dev server."
      />
    );
  }

  return (
    <div className="stack" style={{ gap: "var(--space-md)" }}>
      <div className="stats">
        <Stat
          k="Armed for next run"
          v={num(armedCount)}
          sub={`of ${num(ALL_PROMPTS.length)} in corpus`}
          tone="accent"
        />
        <Stat
          k="Easy"
          v={num(PROMPT_COUNTS.easy)}
          sub={`~${num(meanChars.easy)} chars avg${mix.easy === 0 ? " · weight 0" : ""}`}
        />
        <Stat
          k="Medium"
          v={num(PROMPT_COUNTS.medium)}
          sub={`~${num(meanChars.medium)} chars avg${mix.medium === 0 ? " · weight 0" : ""}`}
        />
        <Stat
          k="Hard"
          v={num(PROMPT_COUNTS.hard)}
          sub={`~${num(meanChars.hard)} chars avg${mix.hard === 0 ? " · weight 0" : ""}`}
        />
      </div>

      <Panel
        title={`Library — ${filtered.length} shown`}
        actions={
          <div className="row">
            <button
              className="btn btn--sm"
              onClick={() =>
                setPromptsEnabled(
                  filtered.map((p) => p.id),
                  true,
                )
              }
            >
              Arm shown
            </button>
            <button
              className="btn btn--sm"
              onClick={() =>
                setPromptsEnabled(
                  filtered.map((p) => p.id),
                  false,
                )
              }
            >
              Disarm shown
            </button>
          </div>
        }
      >
        <div className="row" style={{ marginBottom: "var(--space-sm)" }}>
          <input
            className="input"
            style={{ maxWidth: 280 }}
            placeholder="Search id, title, category, body…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <div className="row" role="group" aria-label="Filter by difficulty">
            {DIFFS.map((d) => (
              <button
                key={d}
                className={`btn btn--sm${diff === d ? "" : " btn--ghost"}`}
                aria-pressed={diff === d}
                onClick={() => setDiff(d)}
              >
                {d}
              </button>
            ))}
          </div>
          <select
            className="select"
            style={{ maxWidth: 200 }}
            value={cat}
            onChange={(e) => setCat(e.target.value)}
            aria-label="Filter by category"
          >
            <option value="all">All categories</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div
          className="scroll-y"
          style={{ maxHeight: "clamp(240px, 46vh, 560px)" }}
        >
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: 40 }}>
                  <span className="sr-only">Armed</span>
                  on
                </th>
                <th style={{ width: 96 }}>id</th>
                <th>title</th>
                <th style={{ width: 150 }}>category</th>
                <th style={{ width: 78 }}>level</th>
                <th style={{ width: 68 }} className="num">
                  chars
                </th>
                <th style={{ width: 74 }} className="num">
                  est. out
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => {
                const on = enabled[p.id] !== false;
                return (
                  <tr
                    key={p.id}
                    data-selected={preview === p.id}
                    onClick={() => setPreview(p.id === preview ? null : p.id)}
                    style={{ cursor: "pointer" }}
                  >
                    <td onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() => togglePrompt(p.id)}
                        aria-label={`Arm ${p.id}`}
                      />
                    </td>
                    <td className="mono dim">{p.id}</td>
                    <td style={{ color: on ? undefined : "var(--color-neutral)" }}>
                      {p.title}
                    </td>
                    <td className="dim">{p.category}</td>
                    <td>
                      <Badge kind={p.difficulty}>{p.difficulty}</Badge>
                    </td>
                    <td className="num">{num(p.text.length)}</td>
                    <td className="num dim">
                      {p.targetTokens ? num(p.targetTokens) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {previewed ? (
        <Panel
          title={`${previewed.id} · ${previewed.category}`}
          actions={
            <button
              className="btn btn--sm btn--ghost"
              onClick={() => setPreview(null)}
            >
              Close
            </button>
          }
        >
          <pre
            className="mono"
            style={{
              margin: 0,
              whiteSpace: "pre-wrap",
              overflowWrap: "anywhere",
              fontSize: "var(--text-2xs)",
              lineHeight: 1.6,
              color: "var(--color-muted)",
              maxHeight: 360,
              overflowY: "auto",
            }}
          >
            {previewed.text}
          </pre>
        </Panel>
      ) : null}
    </div>
  );
}
