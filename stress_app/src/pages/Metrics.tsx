import { useMemo, useState } from "react";
import { useStore } from "../lib/store";
import { clock, mean, ms, num, pct, percentile } from "../lib/format";
import { BarChart, Legend, LineChart, type Series } from "../components/Chart";
import { Empty, Panel, Stat } from "../components/ui";
import type { Difficulty } from "../lib/types";

const C1 = "var(--chart-1)";
const C2 = "var(--chart-2)";
const C3 = "var(--chart-3)";

export default function Metrics() {
  const series = useStore((s) => s.series);
  const completed = useStore((s) => s.completed);
  const metrics = useStore((s) => s.metrics);
  const run = useStore((s) => s.run);
  const [showTable, setShowTable] = useState(false);

  const times = useMemo(() => series.map((s) => s.t), [series]);

  // Everything derived from `completed` needs client traffic to exist; the
  // server-side panels do not, and stay live whether or not a run is going.
  const hasRunData = completed.length > 0;

  const throughput: Series[] = useMemo(() => {
    const out: Series[] = [];
    // Without a run the client line is a row of zeroes, which reads as a
    // measurement rather than the absence of one.
    if (hasRunData) {
      out.push({ name: "client", color: C1, values: series.map((s) => s.outTps) });
    }
    out.push({ name: "server", color: C2, values: series.map((s) => s.serverTps) });
    return out;
  }, [series, hasRunData]);

  // The freshest scrape-derived rate, with a little slack so a poll interval
  // slower than the sampler doesn't blank the tile between ticks.
  const serverTpsNow = useMemo(() => {
    for (const s of series.slice(-3).reverse()) {
      if (s.serverTps != null) return s.serverTps;
    }
    return null;
  }, [series]);

  const ttft: Series[] = useMemo(
    () => [
      { name: "p50", color: C1, values: series.map((s) => s.p50Ttft) },
      { name: "p99", color: C2, values: series.map((s) => s.p99Ttft) },
    ],
    [series],
  );

  const pressure: Series[] = useMemo(
    () => [
      { name: "running", color: C1, values: series.map((s) => s.running) },
      { name: "queued", color: C2, values: series.map((s) => s.queued) },
      { name: "client in-flight", color: C3, values: series.map((s) => s.inFlight) },
    ],
    [series],
  );

  const kv: Series[] = useMemo(
    () => [
      {
        name: "kv cache",
        color: C1,
        values: series.map((s) => (s.kvCache == null ? null : s.kvCache * 100)),
      },
    ],
    [series],
  );

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

  const byDifficulty = useMemo(() => {
    const pick = (d: Difficulty, sel: (v: (typeof ok)[number]) => number | null) => {
      const vals = ok
        .filter((c) => c.difficulty === d)
        .map(sel)
        .filter((v): v is number => v != null);
      return vals.length ? (percentile(vals, 50) as number) : null;
    };
    return {
      ttft: [
        { label: "easy", value: pick("easy", (c) => c.ttftMs), color: C2 },
        { label: "medium", value: pick("medium", (c) => c.ttftMs), color: C1 },
        { label: "hard", value: pick("hard", (c) => c.ttftMs), color: C3 },
      ],
      total: [
        { label: "easy", value: pick("easy", (c) => c.totalMs), color: C2 },
        { label: "medium", value: pick("medium", (c) => c.totalMs), color: C1 },
        { label: "hard", value: pick("hard", (c) => c.totalMs), color: C3 },
      ],
    };
  }, [ok]);

  const elapsedSec = run.elapsedMs / 1000;
  const aggregateTps =
    elapsedSec > 0
      ? ok.reduce((a, b) => a + b.completionTokens, 0) / elapsedSec
      : null;
  const rps = elapsedSec > 0 ? ok.length / elapsedSec : null;
  const errorRate = completed.length
    ? (completed.length - ok.length) / completed.length
    : null;

  if (series.length === 0 && !hasRunData && !metrics) {
    return (
      <Empty
        title="Waiting for the first scrape"
        hint="The vLLM server's Prometheus endpoint is polled once a second, run or no run. If this doesn't clear, check the base URL in Settings."
      />
    );
  }

  return (
    <div className="stack" style={{ gap: "var(--space-md)" }}>
      {hasRunData ? (
        <div className="stats">
          <Stat
            k="Aggregate throughput"
            v={aggregateTps ? aggregateTps.toFixed(0) : "—"}
            unit=" tok/s"
            sub="all workers, run mean"
            tone="accent"
          />
          <Stat
            k="Per-request decode"
            v={tps.length ? (mean(tps) as number).toFixed(1) : "—"}
            unit=" tok/s"
            sub="mean, excludes prefill"
          />
          <Stat
            k="Requests/s"
            v={rps ? rps.toFixed(2) : "—"}
            sub={`${num(ok.length)} succeeded`}
          />
          <Stat k="TTFT p50" v={ms(percentile(ttfts, 50))} sub={`p90 ${ms(percentile(ttfts, 90))}`} />
          <Stat k="TTFT p99" v={ms(percentile(ttfts, 99))} sub={`max ${ms(percentile(ttfts, 100))}`} />
          <Stat
            k="Error rate"
            v={errorRate != null ? pct(errorRate, 1) : "—"}
            sub={`${num(completed.length - ok.length)} failed`}
            tone={errorRate ? "fail" : undefined}
          />
        </div>
      ) : (
        <Panel title="Client-side load">
          <p className="field__hint">
            No requests issued yet — throughput, latency percentiles, and error
            rate fill in once a run starts. The server's own figures below are
            live regardless.
          </p>
        </Panel>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
          gap: "var(--space-md)",
        }}
      >
        <Panel title="Output throughput">
          <LineChart
            series={throughput}
            times={times}
            unit=" tok/s"
            format={(v) => v.toFixed(0)}
          />
          {throughput.length > 1 ? (
            <>
              <div style={{ marginTop: "var(--space-xs)" }}>
                <Legend series={throughput} />
              </div>
              <p className="field__hint" style={{ marginTop: "var(--space-2xs)" }}>
                Client counts a request's tokens when it finishes, so it lags and
                steps; server is vLLM's own counter differenced per scrape, and
                includes traffic from anyone else on the box.
              </p>
            </>
          ) : null}
        </Panel>

        {hasRunData ? (
          <Panel title="Time to first token">
            <LineChart
              series={ttft}
              times={times}
              unit="ms"
              format={(v) => v.toFixed(0)}
            />
            <div style={{ marginTop: "var(--space-xs)" }}>
              <Legend series={ttft} />
            </div>
          </Panel>
        ) : null}

        <Panel title="Server pressure">
          <LineChart series={pressure} times={times} format={(v) => v.toFixed(0)} />
          <div style={{ marginTop: "var(--space-xs)" }}>
            <Legend series={pressure} />
          </div>
        </Panel>

        <Panel title="KV cache utilisation">
          <LineChart
            series={kv}
            times={times}
            unit="%"
            fixedMax={100}
            format={(v) => v.toFixed(0)}
          />
        </Panel>

        {hasRunData ? (
          <>
            <Panel title="Median TTFT by difficulty">
              <BarChart bars={byDifficulty.ttft} unit="ms" format={(v) => v.toFixed(0)} />
            </Panel>

            <Panel title="Median total latency by difficulty">
              <BarChart
                bars={byDifficulty.total}
                unit={"ms"}
                format={(v) => (v >= 10000 ? `${(v / 1000).toFixed(1)}s` : v.toFixed(0))}
              />
            </Panel>
          </>
        ) : null}
      </div>

      <Panel
        title="vLLM server telemetry"
        actions={
          <span
            className={`badge badge--${
              metrics?.ok ? "ok" : metrics ? "fail" : "muted"
            }`}
          >
            {metrics?.ok
              ? "/metrics live"
              : metrics
                ? "/metrics unavailable"
                : "scraping…"}
          </span>
        }
      >
        {metrics?.ok ? (
          <div className="stats">
            <Stat
              k="Generation rate"
              v={serverTpsNow != null ? serverTpsNow.toFixed(0) : "—"}
              unit=" tok/s"
              sub="engine-wide, all clients"
              tone="accent"
            />
            <Stat k="Requests running" v={num(metrics.numRunning)} sub="in the engine batch" />
            <Stat
              k="Requests waiting"
              v={num(metrics.numWaiting)}
              sub="queued for capacity"
              tone={(metrics.numWaiting ?? 0) > 0 ? "fail" : undefined}
            />
            <Stat k="KV cache" v={pct(metrics.kvCacheUsage, 1)} sub="block occupancy" />
            <Stat
              k="Prefix cache hits"
              v={pct(metrics.prefixHitRate, 1)}
              sub="of queried tokens"
            />
            <Stat
              k="Preemptions"
              v={num(metrics.preemptionsTotal)}
              sub="lifetime, engine-wide"
              tone={(metrics.preemptionsTotal ?? 0) > 0 ? "fail" : undefined}
            />
            <Stat
              k="Generated tokens"
              v={num(metrics.generationTokensTotal)}
              sub="lifetime, engine-wide"
            />
            {metrics.specAcceptanceRate != null ? (
              <Stat
                k="Spec-decode accept"
                v={pct(metrics.specAcceptanceRate, 1)}
                sub="drafted tokens kept"
                tone="accent"
              />
            ) : null}
          </div>
        ) : (
          <p className="field__hint">
            {metrics?.error
              ? `Last scrape failed: ${metrics.error}`
              : "Waiting for the first scrape of the Prometheus endpoint."}
          </p>
        )}
      </Panel>

      <Panel
        title="Sampled data"
        actions={
          <button
            className="btn btn--sm btn--ghost"
            onClick={() => setShowTable((v) => !v)}
            aria-expanded={showTable}
          >
            {showTable ? "Hide table" : "Show table"}
          </button>
        }
      >
        {showTable ? (
          <div className="scroll-y" style={{ maxHeight: 320 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>time</th>
                  <th className="num">out tok/s</th>
                  <th className="num">srv tok/s</th>
                  <th className="num">req/s</th>
                  <th className="num">ttft p50</th>
                  <th className="num">ttft p99</th>
                  <th className="num">running</th>
                  <th className="num">queued</th>
                  <th className="num">kv</th>
                </tr>
              </thead>
              <tbody>
                {[...series].reverse().map((s) => (
                  <tr key={s.t}>
                    <td className="dim">{clock(s.t)}</td>
                    <td className="num">{s.outTps.toFixed(0)}</td>
                    <td className="num">{num(s.serverTps)}</td>
                    <td className="num">{s.reqPerSec.toFixed(2)}</td>
                    <td className="num">{ms(s.p50Ttft)}</td>
                    <td className="num">{ms(s.p99Ttft)}</td>
                    <td className="num">{num(s.running)}</td>
                    <td className="num">{num(s.queued)}</td>
                    <td className="num">{pct(s.kvCache, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="field__hint">
            Every point plotted above, as numbers — {num(series.length)} samples
            at one-second resolution.
          </p>
        )}
      </Panel>

      {hasRunData ? (
        <Panel title="Latency distribution">
          <div className="stats">
            <Stat k="Total p50" v={ms(percentile(totals, 50))} />
            <Stat k="Total p90" v={ms(percentile(totals, 90))} />
            <Stat k="Total p99" v={ms(percentile(totals, 99))} />
            <Stat k="Total max" v={ms(percentile(totals, 100))} />
            <Stat k="TTFT mean" v={ms(mean(ttfts))} />
            <Stat k="Sample size" v={num(ok.length)} sub="successful requests" />
          </div>
        </Panel>
      ) : null}
    </div>
  );
}
