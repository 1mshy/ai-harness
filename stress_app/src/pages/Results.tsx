import { useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStore } from "../lib/store";
import type { CompletedRow } from "../lib/types";
import { clock, mean, ms, num, percentile } from "../lib/format";
import { Badge, Empty, Panel, Stat } from "../components/ui";

type SortKey = "seq" | "ttftMs" | "totalMs" | "outputTps" | "completionTokens";
type Outcome = "all" | "ok" | "failed";

export default function Results() {
  const completed = useStore((s) => s.completed);
  const config = useStore((s) => s.config);
  const addLog = useStore((s) => s.addLog);

  const [sort, setSort] = useState<SortKey>("seq");
  const [asc, setAsc] = useState(true);
  const [outcome, setOutcome] = useState<Outcome>("all");
  const [open, setOpen] = useState<string | null>(null);
  const [exporting, setExporting] = useState<string | null>(null);

  const rows = useMemo(() => {
    const filtered = completed.filter((c) =>
      outcome === "all" ? true : outcome === "ok" ? c.ok : !c.ok,
    );
    const dir = asc ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = a[sort] ?? -1;
      const bv = b[sort] ?? -1;
      return (Number(av) - Number(bv)) * dir;
    });
  }, [completed, sort, asc, outcome]);

  const ok = useMemo(() => completed.filter((c) => c.ok), [completed]);
  const ttfts = useMemo(
    () => ok.map((c) => c.ttftMs).filter((v): v is number => v != null),
    [ok],
  );
  const totals = useMemo(() => ok.map((c) => c.totalMs), [ok]);
  const tps = useMemo(
    () => ok.map((c) => c.outputTps).filter((v): v is number => v != null),
    [ok],
  );
  const outTokens = useMemo(
    () => ok.reduce((a, b) => a + b.completionTokens, 0),
    [ok],
  );
  const inTokens = useMemo(
    () => ok.reduce((a, b) => a + b.promptTokens, 0),
    [ok],
  );

  const detail = open ? completed.find((c) => c.id === open) : null;

  async function exportAs(kind: "json" | "csv") {
    setExporting(kind);
    try {
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const filename = `stress-rig-${stamp}.${kind}`;
      const contents =
        kind === "json"
          ? JSON.stringify(
              {
                exportedAt: new Date().toISOString(),
                config,
                summary: {
                  requests: completed.length,
                  succeeded: ok.length,
                  failed: completed.length - ok.length,
                  ttftMs: {
                    p50: percentile(ttfts, 50),
                    p90: percentile(ttfts, 90),
                    p99: percentile(ttfts, 99),
                    mean: mean(ttfts),
                  },
                  totalMs: {
                    p50: percentile(totals, 50),
                    p90: percentile(totals, 90),
                    p99: percentile(totals, 99),
                  },
                  outputTps: { mean: mean(tps), p50: percentile(tps, 50) },
                  promptTokens: inTokens,
                  completionTokens: outTokens,
                },
                requests: completed,
              },
              null,
              2,
            )
          : toCsv(completed);
      const path = await invoke<string>("export_file", { filename, contents });
      addLog("ok", `exported ${completed.length} rows → ${path}`);
    } catch (e) {
      addLog("fail", `export failed · ${String(e)}`);
    } finally {
      setExporting(null);
    }
  }

  if (completed.length === 0) {
    return (
      <Empty
        title="No completed requests"
        hint="Finished requests land here with their timings, token counts, and full response body. Start a run to populate it."
      />
    );
  }

  const th = (key: SortKey, label: string, width?: number) => (
    <th
      data-sortable
      style={width ? { width } : undefined}
      className="num"
      onClick={() => {
        if (sort === key) setAsc((v) => !v);
        else {
          setSort(key);
          setAsc(false);
        }
      }}
    >
      {label}
      {sort === key ? (asc ? " ↑" : " ↓") : ""}
    </th>
  );

  return (
    <div className="stack" style={{ gap: "var(--space-md)" }}>
      <div className="stats">
        <Stat k="Requests" v={num(completed.length)} sub={`${ok.length} ok`} />
        <Stat
          k="Failed"
          v={num(completed.length - ok.length)}
          tone={completed.length - ok.length > 0 ? "fail" : undefined}
          sub={
            completed.length
              ? `${(((completed.length - ok.length) / completed.length) * 100).toFixed(1)}% error rate`
              : undefined
          }
        />
        <Stat k="TTFT p50" v={ms(percentile(ttfts, 50))} sub={`p99 ${ms(percentile(ttfts, 99))}`} />
        <Stat k="Latency p50" v={ms(percentile(totals, 50))} sub={`p99 ${ms(percentile(totals, 99))}`} />
        <Stat
          k="Decode rate"
          v={tps.length ? (mean(tps) as number).toFixed(1) : "—"}
          unit=" tok/s"
          sub="per request, mean"
          tone="accent"
        />
        <Stat
          k="Tokens"
          v={num(outTokens)}
          sub={`${num(inTokens)} prompt in`}
        />
      </div>

      <Panel
        title={`Completed — ${rows.length} shown`}
        actions={
          <div className="row">
            <div className="row" role="group" aria-label="Filter by outcome">
              {(["all", "ok", "failed"] as const).map((o) => (
                <button
                  key={o}
                  className={`btn btn--sm${outcome === o ? "" : " btn--ghost"}`}
                  aria-pressed={outcome === o}
                  onClick={() => setOutcome(o)}
                >
                  {o}
                </button>
              ))}
            </div>
            <button
              className="btn btn--sm"
              onClick={() => void exportAs("json")}
              disabled={exporting !== null}
              data-state={exporting === "json" ? "loading" : undefined}
            >
              Export JSON
            </button>
            <button
              className="btn btn--sm"
              onClick={() => void exportAs("csv")}
              disabled={exporting !== null}
              data-state={exporting === "csv" ? "loading" : undefined}
            >
              Export CSV
            </button>
          </div>
        }
      >
        <div className="scroll-y" style={{ maxHeight: "clamp(240px, 44vh, 520px)" }}>
          <table className="table">
            <thead>
              <tr>
                {th("seq", "#", 56)}
                <th style={{ width: 40 }}>ok</th>
                <th style={{ width: 96 }}>prompt</th>
                <th>title</th>
                <th style={{ width: 76 }}>level</th>
                {th("ttftMs", "ttft", 78)}
                {th("totalMs", "total", 84)}
                {th("completionTokens", "out tok", 76)}
                {th("outputTps", "tok/s", 68)}
                <th style={{ width: 76 }}>finish</th>
                <th style={{ width: 74 }}>at</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.id}
                  data-selected={open === r.id}
                  style={{ cursor: "pointer" }}
                  onClick={() => setOpen(open === r.id ? null : r.id)}
                >
                  <td className="num dim">{r.seq}</td>
                  <td
                    style={{
                      color: r.ok ? "var(--color-ok)" : "var(--color-fail)",
                    }}
                  >
                    {/* Glyph, not colour alone — the hue is reinforcement. */}
                    <span aria-hidden="true">{r.ok ? "✓" : "✕"}</span>
                    <span className="sr-only">{r.ok ? "succeeded" : "failed"}</span>
                  </td>
                  <td className="mono dim">{r.promptId}</td>
                  <td className="truncate" style={{ maxWidth: 260 }}>
                    {r.ok ? r.title : (r.error ?? "failed")}
                  </td>
                  <td>
                    <Badge kind={r.difficulty}>{r.difficulty}</Badge>
                  </td>
                  <td className="num">{ms(r.ttftMs)}</td>
                  <td className="num">{ms(r.totalMs)}</td>
                  <td className="num">{num(r.completionTokens)}</td>
                  <td className="num">
                    {r.outputTps ? r.outputTps.toFixed(1) : "—"}
                  </td>
                  <td className="dim">{r.finishReason ?? "—"}</td>
                  <td className="dim">{clock(r.finishedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {detail ? <DetailPanel row={detail} onClose={() => setOpen(null)} /> : null}
    </div>
  );
}

function DetailPanel({
  row,
  onClose,
}: {
  row: CompletedRow;
  onClose: () => void;
}) {
  return (
    <Panel
      title={`#${row.seq} · ${row.promptId} · ${row.category}`}
      actions={
        <div className="row">
          <button
            className="btn btn--sm btn--ghost"
            onClick={() => void navigator.clipboard.writeText(row.text)}
          >
            Copy response
          </button>
          <button className="btn btn--sm btn--ghost" onClick={onClose}>
            Close
          </button>
        </div>
      }
    >
      <div className="row" style={{ marginBottom: "var(--space-xs)" }}>
        <span className={`badge badge--${row.ok ? "ok" : "fail"}`}>
          {row.ok ? `HTTP ${row.status}` : "failed"}
        </span>
        <span className="badge badge--muted">ttft {ms(row.ttftMs)}</span>
        <span className="badge badge--muted">total {ms(row.totalMs)}</span>
        <span className="badge badge--muted">
          {num(row.promptTokens)} in / {num(row.completionTokens)} out
        </span>
        {row.finishReason ? (
          <span className="badge badge--muted">finish: {row.finishReason}</span>
        ) : null}
      </div>
      <pre
        className="mono"
        style={{
          margin: 0,
          whiteSpace: "pre-wrap",
          overflowWrap: "anywhere",
          fontSize: "var(--text-2xs)",
          lineHeight: 1.6,
          color: row.ok ? "var(--color-ink)" : "var(--color-fail)",
          maxHeight: 380,
          overflowY: "auto",
        }}
      >
        {row.ok ? row.text || "(empty response)" : (row.error ?? "unknown error")}
      </pre>
    </Panel>
  );
}

function toCsv(rows: CompletedRow[]): string {
  const head = [
    "seq",
    "id",
    "promptId",
    "title",
    "category",
    "difficulty",
    "ok",
    "status",
    "ttftMs",
    "totalMs",
    "outputTps",
    "promptTokens",
    "completionTokens",
    "finishReason",
    "worker",
    "finishedAt",
    "error",
  ];
  const esc = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = rows.map((r) =>
    [
      r.seq,
      r.id,
      r.promptId,
      r.title,
      r.category,
      r.difficulty,
      r.ok,
      r.status,
      r.ttftMs ?? "",
      r.totalMs.toFixed(2),
      r.outputTps?.toFixed(3) ?? "",
      r.promptTokens,
      r.completionTokens,
      r.finishReason ?? "",
      r.worker,
      new Date(r.finishedAt).toISOString(),
      r.error ?? "",
    ]
      .map(esc)
      .join(","),
  );
  return [head.join(","), ...lines].join("\n");
}
